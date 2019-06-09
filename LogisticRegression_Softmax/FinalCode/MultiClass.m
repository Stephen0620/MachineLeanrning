function [w,Y,iteration] = MultiClass(X,Label,Lambda,rho,maxIteration)
    K = size(Label,2);
    w = rand(size(X,2),K)';    % Initial Guess
    Y = zeros(size(X,1),K);

    for iteration=1:maxIteration
        wPrev = w;
        for i=1:size(X,1)
            XW = exp(w*X(i,:)');
            Denominator = sum(exp(w*X(i,:)'));
            Y(i,:) = XW./Denominator;        
        end
        for i=1:K
            gradient = sum((Y(:,i)-Label(:,i)).*X)+Lambda.*wPrev(i,:);
            w(i,:) = w(i,:) - rho.*gradient;
        end
        if sum(norm(wPrev-w))<1e-6
            break;
        end
    end    

    for i=1:size(X,1)
        XW = exp(w*X(i,:)');
        Denominator = sum(exp(w*X(i,:)'));
        Y(i,:) = XW./Denominator;        
    end
end

