function [w,Iteration] = GradientDescent(X,Label,lambda,rho,maxIteration)
    w = rand(size(X,2),1);    % Initial Guess
    % gradient = zeros(maxIteration,size(WGradientDescend,1));
    WPrev = w;
    for i=1:maxIteration
        if i>=2&&norm(WPrev)<1e-6
            break;
        else
            WPrev = w;
            XW = X*w;
            y = 1./(1+exp(-XW));
            gradient = sum((y-Label).*X)+lambda.*WPrev';
            w = w - rho.*gradient';
        end
    end
    Iteration = i;
end

