%% Layered architecture for quadrotor control

% Planning with minimum snap 
load('path.mat')
[m, k] = size(path); % number of waypoints
m = 2;
n = 7; % order of polynomial
sigma = {};
sigma.x = [];
sigma.y = [];
sigma.z = [];
sigma.si = [];
sigma.kf = path; % keyframes/waypoints

% Constraints

t = 1e-8:1/(m-1):1.01;

% Values at the keyframes/waypoints are equal to the generated polynomial
% A1 = zeros(m, 4*(n+1)*m);
A1 = [];
for i=1:m-1
    A1 = [A1; zeros(4, 4*(n+1)*(i-1)) reptimeseq(t(i), n) zeros(4, 4*(n+1)*(m-i))];
    A1 = [A1; zeros(4, 4*(n+1)*(i-1)) reptimeseq(t(i+1), n) zeros(4, 4*(n+1)*(m-i))];
end

% Continuity at the keyframes/waypoints
% A2 = zeros(m-1, 4*(n+1)*m);
% A2 = [];
% for i=1:m-2
%     A2 = [A2; zeros(4, 4*(n+1)*(i-1)) reptimeseq(t(i+1), n) zeros(4, 4*(n+1)*(m-i))];
% end

% Derivatives at the end points are 0 (zero velocity, acc, jerk at 0, m)
% A3 = zeros(2, 4*(n+1)*m);
A3 = [];
for j=1:4
    A3 = [A3; derivtimeseq(1, n, j) zeros(4, 4*(n+1)*(m-1))];
    A3 = [A3; zeros(4, 4*(n+1)*(m-1)) derivtimeseq(m, n, j)];
end

% Aeq = [A1; A2; A3];
Aeq = [A1; A3];
beq = [];
for i=1:m-1
    beq = [beq; path(i, :)'; 0];
    beq = [beq; path(i+1, :)'; 0];
end
% beq = reshape([path(1:m, :) zeros(m, 1)], [4*m, 1]);
% beq = [beq; beq(5:end)];
% beq = [beq; beq(5:end); zeros(4*8, 1)]; 
% beq = [beq; beq(2:end); beq(1); beq(m)];
beq = [beq; zeros(4*8, 1)];

% Objective function

% H1 = zeros(m, 3*(n+1)*m);
H1 = [];
for i=1:m-1
    H1 = [H1; zeros(3, 3*(n+1)*(i-1)) snapcomp(t(i+1), t(i), n) zeros(3, 3*(n+1)*(m-i))]; % Fix the indices for t
end
% H1(isnan(H1)) = 1e-8; % check why there are nan values

% H2 = zeros(m, (n+1)*m);
H2 = [];
for i=1:m-1
    H2 = [H2; zeros(1, (n+1)*(i-1)), yawacc(t(i+1), t(i), n), zeros(1, (n+1)*(m-i))];
end

H = [H1'*H1 zeros(3*(n+1)*m, (n+1)*m); zeros((n+1)*m, 3*(n+1)*m) H2'*H2];

% a = zeros(4, 1);
% b = zeros(4, 1);
% 
% for i=1:n
%     a = [a; i*(i-1)*ones(4, 1)];
%     if i < 4
%         b = [b; zeros(4, 1)];
%     else
%         b = [b; i*(i-1)*(i-2)*(i-3)*ones(4, 1)];
%     end
% end
% 
% H = a'*a + b'*b;
% 
% f = [];
% for i=1:m
%     for j=1:n+1
%         f = [f; ones(4, 1)*t(i)^(j-1)];
%     end
% end

opt = quadprog(H, [], [], [], Aeq, beq);

% Generate piecewise polynomial trajectories 
N = 50;
for i=1:m-1
    tsim = t(i):1/(N+1):t(i+1);
    x_idx = 4*(n+1)*(i-1)+1:4:4*(n+1)*i;
    sigma.x = [sigma.x; polyval(opt(x_idx), tsim)];
    polyval(opt(x_idx), 0.01+1/m)
    y_idx = 4*(n+1)*(i-1)+2:4:4*(n+1)*i;
    sigma.y = [sigma.y; polyval(opt(y_idx), tsim)];
    z_idx = 4*(n+1)*(i-1)+3:4:4*(n+1)*i;
    sigma.z = [sigma.z; polyval(opt(z_idx), tsim)];
    si_idx = 4*(n+1)*(i-1)+4:4:4*(n+1)*i;
    sigma.si = [sigma.si; polyval(opt(si_idx), tsim)];
end

%% Centralized Linearized dynamics 
g = 9.80;
m = 1;

   
Ix = 8.1 * 1e-3;
Iy = 8.1 * 1e-3;
Iz = 14.2 * 1e-3;
% 
% A = zeros(12);
% A(1, 2) = 1;
% A(2, 9) = g;
% A(3, 4) = 1;
% A(4, 7) = -g;
% A(5, 6) = 1;
% A(7, 8) = 1;
% A(9, 10) = 1;
% A(11, 12) = 1;
% 
% B = zeros(12, 4);
% B(6, 1) = 1/m;
% B(8, 2) = 1/Ix;
% B(10, 3) = 1/Iy;
% B(12, 4) = 1/Iz;
% 
% % LQR using centralized linearized dynamics
% Q = 100*eye(12);
% R = eye(4);
% N = 25;
% 
% P = dare(A, B, Q, R);
% x0 = 0.1*ones(12, 1); % dim-12 vector
% u = lqr(A, B, Q, R, P, N, x0);
% u = reshape(u, [4, 25]);
% 
% x = zeros(N, 12);
% x(1, :) = A*x0 + B*u(:, 1);
% for i=2:N
%     x(i, :) = A*x(i-1, :)' + B*u(:, i);
% end

% LQR reference tracking 

% MPC reference tracking with state and input constraints


%% Assuming differential flatness

% Decentralized linearized control
% X-subsystem
% The state variables are x, dot_x, pitch, dot_pitch
Ax = [0.0, 1.0, 0.0, 0.0; 
      0.0, 0.0, g, 0.0;
      0.0, 0.0, 0.0, 1.0;
      0.0, 0.0, 0.0, 0.0];
Bx = [0.0; 0.0; 0.0; 1 / Ix];

% Y-subsystem
% The state variables are y, dot_y, roll, dot_roll
Ay = [0.0, 1.0, 0.0, 0.0;
      0.0, 0.0, -g, 0.0;
      0.0, 0.0, 0.0, 1.0;
      0.0, 0.0, 0.0, 0.0];
By = [0.0;
      0.0;
      0.0;
      1 / Iy];

% Z-subsystem
% The state variables are z, dot_z
Az = [0.0, 1.0;
      0.0, 0.0];
Bz = [0.0;
      1 / m];

% Yaw-subsystem
% The state variables are yaw, dot_yaw
Ayaw = [0.0, 1.0;
        0.0, 0.0];
Byaw = [0.0;
        1 / Iz];
    
% LQR reference tracking using error dynamics ei = xi - ri

[Kx, Px] = finite_horizon_dlqr_quad(Ax, Bx, sigma.x); %[sigma.x zeros(1, size(sigma.x, 2)) zeros(1, size(sigma.x, 2)) zeros(1, size(sigma.x, 2))]);

[Ky, Py] = finite_horizon_dlqr_quad(Ay, By, sigma.y);

[Kz, Pz] = finite_horizon_dlqr_quad(Az, Bz, sigma.z);

[Ksi, Psi] = finite_horizon_dlqr_quad(Ayaw, Byaw, sigma.si);

% Control law u = -Kx
% x0 = [];
% x = [];
% for i=1:Nsim
%     u = K()*z;
%     Ax*x + Bx*u
% end

Qx = 1e-4*eye(55);
Qx(1, 1) = 1;
% Qx(2, 2) = 0;
% Qx(3, 3) = 0;
% Qx(4, 4) = 0;

Rx = eye(4);
E1 = zeros(N+1, 4);
E1(1, 1) = 1;
E1(2, 1) = 1;
E1(3, 1) = 1;
E4(4, 1) = 1;

E2 = zeros(N+1, 4);
E2(1, 2) = 1;
E2(2, 2) = 1;
E2(3, 2) = 1;
E2(4, 2) = 1;
Z = circshift(eye(N+1), 1, 2);
Ax = eye(4);
Bx = eye(4);
Atilde = [Ax Ax*E1' - E2'; zeros(N+1, 4) Z]; % Augmented state zi = [ei; \Bar{ri}]'
Btilde = [Bx; zeros(N+1, 4)];


% % Check stabilizability
% lambdaA = eig(Atilde);
% 
% for i=1:size(lambdaA)
%     abs(lambdaA(i))
% end
% 
% % Check detectability
% 
% [K, S, e] = dlqr(Atilde, Btilde, Qx, Rx, zeros(55, 4));
% Px = idare(Atilde, Btilde, Qx, Rx, [], []);
% % Px = dlyap
% u = lqr(Atilde, Btilde, Qx, Rx, Px, 10, zeros(4, 1));

    


