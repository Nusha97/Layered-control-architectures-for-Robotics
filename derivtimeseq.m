function b = derivtimeseq(t, n, j)
    b = [];
    switch j
        case 1
            for i=1:n+1
                b = [b (i-1)*t^(i-2)*eye(4)];
            end
        case 2
            for i=1:n+1
                b = [b (i-1)*(i-2)*t^(i-3)*eye(4)];
            end
        case 3
            for i=1:n+1
                b = [b (i-1)*(i-2)*(i-3)*t^(i-4)*eye(4)];
            end
        case 4
            for i=1:n+1
                b = [b (i-1)*(i-2)*(i-3)*(i-4)*t^(i-5)*eye(4)];
            end   
   end
end