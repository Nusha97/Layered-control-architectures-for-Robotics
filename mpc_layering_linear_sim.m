%% Layered architecture for quadrotor control

% Planning with minimum snap 
[m, k] = size(path); % number of waypoints
n = 5; % order of polynomial
sigma = {};
sigma.x = zeros(m, n);
sigma.y = zeros(m, n);
sigma.z = zeros(m, n);
sigma.si = zeros(m, n);
sigma.kf = path;

% Constraints

t = 1:m;

% Values at the keyframes/waypoints are equal to the generated polynomial
% A1 = zeros(m, 4*(n+1)*m);
A1 = [];
for i=1:m
    A1 = [A1; zeros(4, 4*(n+1)*(i-1)) reptimeseq(t(i), n) zeros(4, 4*(n+1)*(m-i))];
end

% Continuity at the keyframes/waypoints
% A2 = zeros(m-1, 4*(n+1)*m);
A2 = [];
for i=2:m
    A2 = [A2; zeros(4, 4*(n+1)*(i-2)) reptimeseq(t(i), n) -reptimeseq(t(i), n) zeros(4, 4*(n+1)*(m-i))];
end

% Derivatives at the end points are 0 (zero velocity, acc, jerk at 0, m)
% A3 = zeros(2, 4*(n+1)*m);
A3 = [];
A3 = [A3; derivtimeseq(1, n) zeros(4, 4*(n+1)*(m-1))];
A3 = [A3; zeros(4, 4*(n+1)*(m-1)) derivtimeseq(m, n)];


Aeq = [A1; A2; A3];
beq = reshape([path zeros(m, 1)], [4*m, 1]);
beq = [beq; beq(5:end); beq(1:4); beq(21:end)]; 
% beq = [beq; beq(2:end); beq(1); beq(m)];

% Objective function

% H1 = zeros(m, 3*(n+1)*m);
H1 = [];
for i=1:m
    H1 = [H1; zeros(3, 3*(n+1)*(i-1)) snapcomp(t(i), t(i)-1, n) zeros(3, 3*(n+1)*(m-i))]; % Fix the indices for t
end
H1(isnan(H1)) = 0; % check why there are nan values

% H2 = zeros(m, (n+1)*m);
H2 = [];
for i=1:m
    H2 = [H2; zeros(1, (n+1)*(i-1)), yawacc(t(i), t(i)-1, n), zeros(1, (n+1)*(m-i))];
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

tsim = 1:1/(n+1):m;
% Reference trajectories
for i=1:m
    
end

% Centralized Linearized dynamics 
g = 9.80;
m = 1;

   
Ix = 8.1 * 1e-3;
Iy = 8.1 * 1e-3;
Iz = 14.2 * 1e-3;

A = zeros(12);
A(1, 2) = 1;
A(2, 9) = g;
A(3, 4) = 1;
A(4, 7) = -g;
A(5, 6) = 1;
A(7, 8) = 1;
A(9, 10) = 1;
A(11, 12) = 1;

B = zeros(12, 4);
B(6, 1) = 1/m;
B(8, 2) = 1/Ix;
B(10, 3) = 1/Iy;
B(12, 4) = 1/Iz;

% LQR using centralized linearized dynamics
Q = 100*eye(12);
R = eye(4);
N = 25;

P = dare(A, B, Q, R);
x0 = 0.1*ones(12, 1); % dim-12 vector
u = lqr(A, B, Q, R, P, N, x0);
u = reshape(u, [4, 25]);

x = zeros(N, 12);
x(1, :) = A*x0 + B*u(:, 1);
for i=2:N
    x(i, :) = A*x(i-1, :)' + B*u(:, i);
end

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
    
% LQR reference tracking using error dynamics

% E1 = ;
% E2 = ;
% Atilde = [Ax Ax*E1' - E2; zeros() Z]% Augmented state zi = [ei; \Bar{ri}]'
    
% Reference signal 

