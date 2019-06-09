function [w,y] = MiniBatch(X,Label,Lambda,Epoch,BatchSize,Beta)
    rho = 1/(size(X,1));
    w = zeros(size(X,2),1); %Initial guess 
    
    Batch = BatchSize;
    for j=1:Epoch
        idx = randperm(size(X,1));
        XRandom = X(idx,:);
        tRandom = Label(idx,:);
        for i=1:Batch:size(X,1)
            XW = XRandom(i:i+Batch-1,:)*w;
            y = 1./(1+exp(-XW));
            gradient = sum((y-tRandom(i:i+Batch-1)).*XRandom(i:i+Batch-1,:))+Lambda.*w';

            % Momentum
            if i==1
                V = (1-Beta).*gradient;
            else
                V = Beta.*V+(1-Beta).*gradient;
            end
            w = w-rho.*V';   % Update w
        end
    end
    XW = X*w;
    y = 1./(1+exp(-XW));
end

