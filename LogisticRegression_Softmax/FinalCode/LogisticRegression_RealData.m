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
X = [D(:,4:6),ones(size(D,1),1)]; % Designed matrix

%% Logistic Regression using Gradient Descend
rho = 0.05;
maxIteration = 100000;
Lambda=0;
disp('Training time of Logistic Regression on real data:...')
tic
[w,Y,Iteration] = MultiClass(X,T,Lambda,rho,maxIteration);
toc

%% Predict
[~,P] = max(Y,[],2);
