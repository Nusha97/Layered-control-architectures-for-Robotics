function a = reptimeseq(t, n)
    a = [];
    for i=1:n+1
        a = [a t^(i-1)*eye(4)];
    end
end