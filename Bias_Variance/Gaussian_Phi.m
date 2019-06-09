function Phi = Gaussian_Phi(Target,X,s)
    % Make sure the target matrix is a column vector
    if size(Target,1)==1
        Target=Target';
    end
    
    N = numel(Target);
    Phi = zeros(N,numel(X)-1);
    for i=1:N
        x = ones(1,numel(X)-1);
        M = X(1:numel(X)-1)';   % 1~N-1
        x = x.*Target(i);
        x = (-(x-M).^2)./(2*(s^2));
        Phi(i,:) = exp(x);
    end
    Phi = [ones(N,1),Phi];
end

