%% Layered architecture for quadrotor control

% Planning with minimum snap 



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
% Reference signal 

