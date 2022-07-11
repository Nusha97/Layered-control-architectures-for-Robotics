function b = derivtimeseq(t, n)
    b = [];
    for i=1:n+1
        b = [b (i-1)*t^(i-2)*eye(4)];
    end
end