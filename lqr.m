function u_lqr = lqr(A, B, Q, R, P, N, x_0)
    S_x = eye(2);
    for i=1:N
        S_x = [S_x; A^i];
    end

    S_u = zeros(2, N);
    k = N-1;
    for i = 1:N
        S_u = [S_u; cont(A, B, i) zeros(2, k)];
        k = k-1;
    end

    Q_hat = kron(eye(N), Q);
    Q_hat = [Q_hat zeros(2*N, 2); zeros(2, 2*N), P]; 

    R_hat = kron(eye(N), R);
    
    %% Solution from quadratic program optimization

    H = S_u'*Q_hat*S_u + R_hat;
    F = S_x'*Q_hat*S_u;

    f = 2*F'*x_0;

    u_lqr = quadprog(2*H, f);

%    J_0_opt = U_0_opt'*H*U_0_opt + 2*x_0'*F*U_0_opt + x_0'*S_x'*Q_hat*S_x*x_0;
end