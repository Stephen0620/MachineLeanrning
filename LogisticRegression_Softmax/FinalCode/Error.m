function ErrorRate = Error(TrueLabel,FitLabel)
    if size(TrueLabel)~=size(FitLabel)
        FitLabel = FitLabel';
    end
    Compare = TrueLabel==FitLabel;
    Error = numel(find(Compare==0));
    ErrorRate = Error/numel(TrueLabel);
end

