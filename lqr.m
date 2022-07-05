function u_lqr = lqr(A, B, Q, R, P, N, x_0)
    dimA = size(A);
    S_x = eye(dimA(1));
    for i=1:N
        S_x = [S_x; A^i];
    end
    
    dimB = size(B);
    S_u = zeros(dimB(1), dimB(2)*N);
    k = N-1;
    for i = 1:N
        S_u = [S_u; cont(A, B, i) zeros(dimB(1), dimB(2)*k)];
        k = k-1;
    end

    Q_hat = kron(eye(N), Q);
    Q_hat = [Q_hat zeros(dimA(1)*N, dimA(2)); zeros(dimA(2), dimA(1)*N), P]; 

    R_hat = kron(eye(N), R);
    
    %% Solution from quadratic program optimization

    H = S_u'*Q_hat*S_u + R_hat;
    F = S_x'*Q_hat*S_u;

    f = 2*F'*x_0;

    u_lqr = quadprog(2*H, f);

%    J_0_opt = U_0_opt'*H*U_0_opt + 2*x_0'*F*U_0_opt + x_0'*S_x'*Q_hat*S_x*x_0;
end