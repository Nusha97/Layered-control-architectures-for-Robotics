function snap = snapcomp(t1, t2, n)
    snap = [];
    for j=1:n+1
        snap = [snap j*(j-1)*(j-2)*(t1^(j-3)-t2^(j-3))*eye(3)];
    end
end