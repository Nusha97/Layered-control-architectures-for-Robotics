function r = yawacc(t1, t2, n)
    r = [];
    for i=1:n+1
        r = [r i*(t1^(i-1)-t2^(i-1))];
    end
end