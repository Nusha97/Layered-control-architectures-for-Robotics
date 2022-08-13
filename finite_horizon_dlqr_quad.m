function [M, F] = finite_horizon_dlqr_quad(A, B, v)

%horizons
N = 52;
H = floor(N)+1;
t = 1:N;

%tracking penalty weight
rho = 0.5;

%augmented state space to do reference tracking
n = size(A,1);
p = size(B,2);

e1 = zeros(n*H,n);
e2 = zeros(n*H,n);

e1(1:n,1:n) = eye(n);
e2(n+1:2*n,1:n) = eye(n);

Z = zeros(n*H,n*H);

shift = blkdiag(eye(n),eye(n));
for k = 1:H-3
    shift = blkdiag(shift,eye(n));
end

Z(1:end-n,n+1:end) = shift;


bA = [A, A*e1'-e2'; zeros(n*H,n),Z];
bB = [B;zeros(n*H,p)];

%Riccati recursion.  Here Q = tracking error penalty, R is control penalty
M{N+1} = rho*blkdiag(eye(n),zeros(n*H,n*H));
Q = M{N+1};
R = 1000*eye(p);
for k = N:-1:1
    M{k} = Q + bA'*M{k+1}*bA - bA'*M{k+1}*bB*inv(R+bB'*M{k+1}*bB)*bB'*M{k+1}*bA;
    F{k} =  inv(R+bB'*M{k+1}*bB)*bB'*M{k+1}*bA;
end
% R = 1;


%% closed form, only optimize over reference trajectory
cvx_begin
cvx_solver sdpt3
expression x0(n,1);

variable rp(n,H);

%initial tracking error; reference trajectory
x0 = [zeros(n, 1)-rp(:,1);vec(rp)];

%cost function components

func_util = norm(rp(1,1:N)-v(1,1:N),1);
tracking_penalty = x0'*M{1}*x0;
minimize(func_util + tracking_penalty);
subject to
% reference constraints
%  virtual dynamics: let's make r(1,:) behave like a single integrator being
% driven by r(2,:), and the rest is free
for k = 1:N
    rp(1,k+1) == rp(1,k)+5*rp(2,k);
end
%reference constraints
%and let's constrain our virtual velocity
% rp(1,:) == rp(2,:);
% rp(2,:) == rp(4,:);
% rp(4,:) == 0;
% -pi/5 <= rp(3,:) <= pi/5;
% -pi/5 <= rp(4,:) <= pi/5;
cvx_end
 
closed_form_cost = func_util + tracking_penalty;
closed_form_cost

close all

%% Quadrotor trajectory simulation

z = zeros(n,N+1);
z(:,1) = zeros(n, 1);
for k = 1:N
    rk = Z^(k-1)*vec(rp);
    u(:,k) = -F{k}*[z(:,k);rk];
    z(:,k+1) = A*z(:,k) + B*u(:,k);
end

figure();
plot(1:N,z(1,1:N),'b-',1:N,rp(1,1:N),'g-',1:N,v(1,1:N),'r--','LineWidth',1)
legend('linear state','reference','min snap ref');
title(sprintf('rho = %0.2f',rho));
set(gca,'FontSize',16,'fontWeight','bold')
set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
xlabel("time");
ylabel("state");

%% Pendulum parameters
% M = 2;
% m = 0.2;
% b = 0.1;
% I = 0.006;
% g = 9.8;
% l = 1;

% tau = 0.1;
% %% simulation based on previously computed reference
% z = zeros(n,N+1);
% nl = zeros(n,10*(N+1));
% z(:,1) = [0;0;0;0];
% nl(:,1) = [0;0;pi;0];
% 
% var = 0.0;
% w1k = var*randn(2,1);
% w2k = var*randn(2,1);
% randn('seed',0);
% for k = 1:N
%     rk = Z^(k-1)*vec(rp);
%     u(:,k) = -F{k}*[z(:,k)-rp(:,k);rk];
%     w1k = var*randn(1,1);
%     w2k = var*randn(1,1);
%     z(:,k+1) = A*z(:,k) + B*u(:,k)+[0;w1k;0;w2k];
% end
% 
% var = 0.0;
% w1k = var*randn(2,1);
% w2k = var*randn(2,1);
% randn('seed',0);
% factor = 20;
% tau = tau/factor;
% 
% %%nonlinear model
% %xddot = (u - b*xdot + l*m*thetadot^2*sin(theta) + (g*l^2*m^2*cos(theta)*sin(theta))/(m*l^2 + I))/(M + m - (l^2*m^2*cos(theta)^2)/(m*l^2 + I));
% %thetaddot =-(l*m*(l*m*cos(theta)*sin(theta)*thetadot^2 + u*cos(theta) -b*xdot*cos(theta) + g*m*sin(theta) + M*g*sin(theta)))/(I*m + I*M + l^2*m^2 - l^2*m^2*cos(theta)^2 + M*l^2*m);
%  
% for t = 1:N*factor
%     k = ceil(t/factor);
%     rk = Z^(k-1)*vec(rp);
%     uu(:,t) = -F{k}*[nl(1:2,t)-rp(1:2,k);
%                      nl(3,t)-pi-rp(3,k);
%                      nl(4,t)-rp(4,k);
%                      rk];
%     w1k = var*randn(1,1);
%     w2k = var*randn(1,1);
% %     if (mod(t-1,factor)==0)
% %         ctrl = u(:,k)
% %         ctrlnl = uu(:,max(1,t-1))
% %     end
%     x = nl(1,t);
%     xdot = nl(2,t);
%     theta = nl(3,t);
%     thetadot = nl(4,t);
%     xddot = (uu(:,t) - b*xdot + l*m*thetadot^2*sin(theta) + (g*l^2*m^2*cos(theta)*sin(theta))/(m*l^2 + I))/(M + m - 0*(l^2*m^2*cos(theta)^2)/(m*l^2 + I));
%     thetaddot =-(l*m*(l*m*cos(theta)*sin(theta)*thetadot^2 + uu(:,t)*cos(theta) -b*xdot*cos(theta) + g*m*sin(theta) + M*g*sin(theta)))/(I*m + I*M + l^2*m^2 - 0*l^2*m^2*cos(theta)^2 + M*l^2*m);
% %     
% %     
% %     xddot
% %     1/(M+m)*(-b*xdot+uu(:,t))
% 
%     nl(:,t+1) = [0;w1k;0;w2k]+...
%                 [x+tau*xdot;
%                 xdot+tau*xddot;
%                 theta+tau*thetadot;
%                 thetadot+tau*thetaddot;];
% end
% % raw_cost
% % solved_cost
% 
% 
% 
% figure
% subplot(2,2,1);
% plot(1:N,z(1,1:N),'b-',1:N,nl(1,1:factor:N*factor),'r-',1:N,rp(1,1:N),'k:',1:N,v(1,1:N),'g-','LineWidth',4)
% % legend('linear state x1(1)','nonlinear state x1(1)','reference r1(1)','ideal trajectory v');%,'Location','SouthEast')
% title(sprintf('rho = %0.2f',rho));
% set(gca,'FontSize',16,'fontWeight','bold')
% set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
% 
% subplot(2,2,2);
% plot(1:N,z(2,1:N),'b-',1:N,nl(2,1:factor:N*factor),'r-',1:N,rp(2,1:N),'k:','LineWidth',4)
% % legend('linear state x1(2)','nonlinear state x1(2)','reference r1(2)');%,'Location','SouthEast')
% title(sprintf('rho = %0.2f',rho));
% set(gca,'FontSize',16,'fontWeight','bold')
% set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
% 
% subplot(2,2,3);
% plot(1:N,pi+z(3,1:N),'b-',1:N,nl(3,1:factor:N*factor),'r-',1:N,pi+rp(3,1:N),'k:','LineWidth',4)
% % legend('linear state x1(3)','nonlinear state x1(3)','reference r1(3)');%,'Location','SouthEast')
% title(sprintf('rho = %0.2f',rho));
% set(gca,'FontSize',16,'fontWeight','bold')
% set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')
% 
% subplot(2,2,4);
% plot(1:N,z(4,1:N),'b-',1:N,nl(4,1:factor:N*factor),'r-',1:N,rp(4,1:N),'k:','LineWidth',4)
% % legend('linear state x1(2)','nonlinear state x1(2)','reference r1(2)');%,'Location','SouthEast')
% title(sprintf('rho = %0.2f',rho));
% set(gca,'FontSize',16,'fontWeight','bold')
% set(findall(gcf,'type','text'),'FontSize',16,'fontWeight','bold')

