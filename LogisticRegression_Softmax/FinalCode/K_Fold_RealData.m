clear,clc;
close all;

%% Read the Leaf data
L = textread('Type.txt','%s');
D = xlsread('Data.xls');
Type = unique(L);
K = numel(Type);

for i=1:K
    idx = find(strcmp(L,Type{i})==1);
    T(idx,i) = 1;
    Label(idx) = i;
end

% Normalize
D = (D - mean(D))./std(D);
% Designed Matrix
X = [D(:,4:6),ones(size(D,1),1)]; 

% Seperate the data into training set and testing set
k = 4;
Indices = crossvalind('Kfold', size(X,1), 4);

%% K-Fold training
w = cell(k,1);
Y = cell(k,1);
Iteration = zeros(k,1);
ErrorRate = zeros(k,1);
P = cell(k,1);

for i=1:k
    % Training set
    idx = find(Indices~=i);
    X_train = X(idx,:);
    T_train = T(idx,:);
    
    % Training using gradient descent
    rho = 0.05;
    maxIteration = 100000;
    Lambda = 0;
    
    disp('Training Time of gradient descent:');
    tic
    [w{i},~,Iteration(i)] = MultiClass(X_train,T_train,Lambda,rho,maxIteration);
    toc
    
    % Testing set
    idx = find(Indices==i);
    X_test = X(idx,:);
    Label_test = Label(idx);
    
    % Get the prediction
    XW = exp(w{i}*X_test');
    Denominator = sum(exp(w{i}*X_test'));
    Y_test = XW./Denominator; 
    [~,P{i}] = max(Y_test',[],2);
    
    % Calculate the Error Rate
    ErrorRate(i) = Error(Label_test,P{i});
end

%% Error rate of K-fold
Error = mean(ErrorRate);
