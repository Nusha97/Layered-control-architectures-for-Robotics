%% Layered architecture for quadrotor control

% Planning with minimum snap 

m = 7; % number of waypoints
n = 5; % order of polynomial
sigma = {};
sigma.x = zeros(m, n);
sigma.y = zeros(m, n);
sigma.z = zeros(m, n);
sigma.si = zeros(m, n);
sigma.kf = path;

% Constraints

a = zeros(4, 1);
b = zeros(4, 1);

for i=1:n
    a = [a; i*(i-1)*ones(4, 1)];
    if i < 4
        b = [b; ones(4, 1)];
    else
        b = [b; i*(i-1)*(i-2)*(i-3)];
    end
end

H = a'*a + b'*b;

t = 1:m;

f = [];
for i=1:m
    for j=1:n+1
        f = [f; ones(4, 1)*t(i)^(j-1)];
    end
end

opt = quadprog(H, [], [], [], blkdiag(ones(4*(n+1)*m, 1)), f);

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

