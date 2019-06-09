function [w,y] = Stochastic(X,Label,Lambda,Epoch)
    rho = 1/(size(X,1));
    w = zeros(size(X,2),1); %Initial guess 
    idx = randperm(size(X,1));
    XRandom = X(idx,:);
    tRandom = Label(idx,:);

    for j=1:Epoch
        for i=1:size(X,1)
            y = 1./(1+exp(-XRandom(i,:)*w));
            w = w-((rho).*(y-tRandom(i)).*XRandom(i,:))'-rho.*Lambda.*w;
        end
    end
    XW = X*w;
    y = 1./(1+exp(-XW));
end

