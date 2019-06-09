function Phi = Polynomial_Phi(X,M)
    if (M==0)
        Phi=ones(numel(X),1);
    else
        for i=1:numel(X)
            for j=1:M
                Phi(i,j)=X(i)^j;
            end
        end
        Phi=[ones(numel(X),1),Phi];
    end
end

