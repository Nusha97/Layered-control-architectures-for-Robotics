function u = mympc(A, B, Q, R, P, N, umin, umax, xmin, xmax, x)
    %% State constraints 
    Au = [1 -1]';
    bu = [umax -umin]';
    Ax = [0 -1];
    G_0 = kron(eye(N), Au);
    
    G_0 = [G_0; zeros(1, N); Ax*B zeros(1, N-1)];
    k = N-2;
    for i = 2:N
        G_0 = [G_0; Ax*cont(A, B, i) zeros(1, k)];
        k = k-1;
    end

    E_0 = [zeros(2*N, 2); -Ax];
%     E_0 = [];
    for i=2:N+1
        E_0 = [E_0; -Ax*A^(i-1)];
    end
    
    disp(E_0);

    b_x = -xmin(2);
    w_0 = [repmat(bu, N, 1); repmat(b_x, N+1, 1)];
    disp(w_0);

    A_ineq = G_0;
    disp(A_ineq);
    b_ineq = w_0 + E_0*x;
    disp(b_ineq);
    
    %% Calculating cost matrices
    A_hat = eye(2);
    for i=1:N
        A_hat = [A_hat; A^i];
    end
    
    B_hat = zeros(2, N);
    k = N-1;
    for i = 1:N
        B_hat = [B_hat; cont(A, B, i) zeros(2, k)];
        k = k-1;
    end

    Q_hat = kron(eye(N), Q);
    Q_hat = [Q_hat zeros(2*N, 2); zeros(2, 2*N), P]; 

    R_hat = kron(eye(N), R);
    
    %% Quadratic Program
    
    H = 2 * (B_hat'*Q_hat*B_hat + R_hat);
    F = 2*A_hat'*Q_hat*B_hat;

    f = F'*x;

    if xmin(2) == -Inf
        u = quadprog(H, f, [], [], [], [], umin*ones(N,1), umax*ones(N,1)); % Without state constraints 
    else
        u = quadprog(H, f, A_ineq, b_ineq, [], [], umin*ones(N,1), umax*ones(N,1));
    end
end