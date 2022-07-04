function mat = cont(A, B, n)
mat = [];
    for i=1:n
        mat = [A^(i-1)*B mat];
    end
end