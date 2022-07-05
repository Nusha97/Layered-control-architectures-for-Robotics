%% Main

A = [0.7115 -0.4345; 0.4345 0.8853];
B = [0.2173; 0.0573];
C = [0 1];

umin = -5;
umax = 5;

x_0 = [0; 10];

t = 0:0.1:5;
U = zeros(size(t));

Q = 10*eye(2);
R = 1;

N = 5;

%% Calculating terminal weight matrices

P_are = dare(A, B, Q, R);
P_Lyap = dlyap(A, Q);

%% MPC control

F_are = -inv(B'*P_are*B+R)*B'*P_are*A;

x_are = zeros(2, N);

k = 1;
x_are(:, k) = x_0;
u = mympc(A, B, Q, R, P_are, N, umin, umax, [-Inf; -Inf], [Inf; Inf], x_are(:, k));
U(k) = u(1);

for j = 0.1:0.1:5

    x_are(:, k+1) = A*x_are(:, k) + B*U(k);
    u = mympc(A, B, Q, R, P_are, N, umin, umax, [-Inf; -Inf], [Inf; Inf], x_are(:, k+1));
    k = k+1;
    U(k) = u(1);
    
%     k = k+1;
end

figure();
plot(t, x_are(1, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 1 for N=5');
xlabel('Time (s)');
ylabel('x(1)');

 
figure();
plot(t, x_are(2, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 2 for N=5');
xlabel('Time (s)');
ylabel('x(2)');

%% Horizon N=1

P_sh = Q; % small horizon
N = 1;

F_sh = -inv(B'*P_sh*B+R)*B'*P_sh*A;

x_sh = zeros(2, N);

k = 1;
x_sh(:, k) = x_0;
u = mympc(A, B, Q, R, P_sh, N, umin, umax, [-Inf; -Inf], [Inf; Inf], x_sh(:, k));
U(k) = u(1);

for j = 0.1:0.1:5

    x_sh(:, k+1) = A*x_sh(:, k) + B*U(k);
    u = mympc(A, B, Q, R, P_sh, N, umin, umax, [-Inf; -Inf], [Inf; Inf], x_sh(:, k+1));
    k = k+1;
    U(k) = u(1);
    
end

figure();
plot(t, x_sh(1, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 1 for N=1');
xlabel('Time (s)');
ylabel('x(1)');

 
figure();
plot(t, x_sh(2, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 2 for N=1');
xlabel('Time (s)');
ylabel('x(2)');

%% LQR optimal controller

Q_lqr = 100 * eye(2);
R_lqr = 1;
N = 2;

U_lqr = zeros(4, N);
U_mpc = zeros(4, N);

% Initial state [0 10]'
x_0 = [0, 10]';

% Using the terminal weight as the solution of ARE
P_lqr = dare(A, B, Q_lqr, R_lqr);

u = lqr(A, B, Q_lqr, R, P_lqr, N, x_0);
U_lqr(1, :) = u;

u = mympc(A, B, Q_lqr, R_lqr, P_lqr, N, umin, umax, [-Inf, -Inf]', [Inf, Inf]', x_0);
U_mpc(1, :) = u;

% Using the terminal weight as the solution of Lyapunov equation
P_L = dlyap(A, Q_lqr);

u = lqr(A, B, Q_lqr, R, P_L, N, x_0);
U_lqr(2, :) = u;

u = mympc(A, B, Q_lqr, R_lqr, P_L, N, umin, umax, [-Inf, -Inf]', [Inf, Inf]', x_0);
U_mpc(2, :) = u;

% Initial state [0.1 0.1]'
x_0 = [0.1, 0.1]';

% Using the terminal weight as the solution of ARE

u = lqr(A, B, Q_lqr, R, P_lqr, N, x_0);
U_lqr(3, :) = u;

u = mympc(A, B, Q_lqr, R_lqr, P_lqr, N, umin, umax, [-Inf, -Inf]', [Inf, Inf]', x_0);
U_mpc(3, :) = u;

% Using the terminal weight as the solution of Lyapunov equation
P_L = dlyap(A, Q_lqr);

u = lqr(A, B, Q_lqr, R, P_L, N, x_0);
U_lqr(4, :) = u;

u = mympc(A, B, Q_lqr, R_lqr, P_L, N, umin, umax, [-Inf, -Inf]', [Inf, Inf]', x_0);
U_mpc(4, :) = u;

%% MPC using state constraints

Q = eye(2);
R = 1;
N = 5;
P = dare(A, B, Q, R);
U_s = zeros(1, 51);

xmin = [-Inf, 0]';
xmax = [Inf, Inf]';

x_0 = [0 6]';

F_are = -inv(B'*P*B+R)*B'*P*A;

x_are = zeros(2, N);

k = 1;
x_are(:, k) = x_0;
u = mympc(A, B, Q, R, P, N, umin, umax, xmin, xmax, x_are(:, k));
U_s(k) = u(1);

for j = 0.1:0.1:5

    x_are(:, k+1) = A*x_are(:, k) + B*U_s(k);
    u = mympc(A, B, Q, R, P, N, umin, umax, xmin, xmax, x_are(:, k+1));
    k = k+1;
    U_s(k) = u(1);
    
end

figure();
plot(t, x_are(1, :), 'LineWidth', 1.5);
hold on
plot(t, U_s, 'LineWidth', 1.5);
title('Evolution of state 1 for N=5');
xlabel('Time (s)');
ylabel('x(1)');

 
figure();
plot(t, x_are(2, :), 'LineWidth', 1.5);
hold on
plot(t, U_s, 'LineWidth', 1.5);
title('Evolution of state 2 for N=5');
xlabel('Time (s)');
ylabel('x(2)');

x_0 = [0 10]';

F_are = -inv(B'*P*B+R)*B'*P*A;

x_are = zeros(2, N);

k = 1;
x_are(:, k) = x_0;
u = mympc(A, B, Q, R, P, N, umin, umax, xmin, xmax, x_are(:, k));
U(k) = u(1);

for j = 0.1:0.1:5

    x_are(:, k+1) = A*x_are(:, k) + B*U(k);
    u = mympc(A, B, Q, R, P, N, umin, umax, xmin, xmax, x_are(:, k+1));
    k = k+1;
    U(k) = u(1);
    
end

figure();
plot(t, x_are(1, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 1 for N=5');
xlabel('Time (s)');
ylabel('x(1)');

 
figure();
plot(t, x_are(2, :), 'LineWidth', 1.5);
hold on
plot(t, U, 'LineWidth', 1.5);
title('Evolution of state 2 for N=5');
xlabel('Time (s)');
ylabel('x(2)');



