function R = ReLU(X,Option)
    if strcmp(Option,'Foward')
        R = X;
        R(X<=0) = 0;
    elseif strcmp(Option,'Back')
        R = X;
        R(X>0) = 1;
        R(X<=0) = 0;
    end
end

