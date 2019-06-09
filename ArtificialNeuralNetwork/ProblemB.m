clear,clc;
close all;

rng(100);
%% Synthetic data
N = 100;
X=2*rand(1,N)-1;
t=sin(2*pi*X)+0.3*randn(1,N);

D = size(X,1);
Layers = [D,3,1]; % Number of units in each layers, input , hidden and output

figure;
scatter(X,t);
xlabel('Data');
ylabel('t');

%Activation Function
H{1} = @(x) tanh(x);
% H{2} = @(x) tanh(x);
H{2} = @(x) x;  % Output layer activation function
dH{1} = @(x) 1 - tanh(x).^2;
% dH{2} = @(x) 1 - tanh(x).^2;
dH{2} = @(x) 1;

%% Feed Forward Network
Z{1} = X;
W = cell(numel(Layers)-1,1);    % Weight Vector
B = cell(numel(Layers)-1,1);    % Bias term
Delta = cell(numel(Layers)-1,1);
rho = 0.01;

% Initial Guess Might need to be revised
for i=1:numel(W)
%     W{i} = 0.01+(0.1-0.01)*rand(Layers(i+1),Layers(i));
%     b = 0.01+(0.1-0.01)*rand(Layers(i+1),1);
    W{i} = rand(Layers(i+1),Layers(i));
    b = zeros(Layers(i+1),1);
    B{i} = repmat(b,[1,size(X,2)]);
end

Epoch = 8000;
for j=1:Epoch
    % Feed Forward
    for i=1:numel(Layers)-1
        A{i} = W{i}*Z{i}+B{i};
        h = H{i};
        Z{i+1} = h(A{i});
    end
    y = Z{numel(Layers)};
    
    % Back propagation
    Delta{end} = (y-t); % Output layer cost derivative
%     E(j) = sum(-t.*log(y)-(1-t).*log(1-y)); % Cost function
    E(j) = sum((y-t).^2); % Cost function
    for i=numel(Layers)-1:-1:1
        if i==1
            break;
        else
            dh = dH{i-1};
            Delta{i-1}  = W{i}'*Delta{i}.*dh(A{i-1});    % Delta(1)
        end
    end    
    for i=numel(Layers)-1:-1:1
        W{i} = W{i}-rho.*(Delta{i}*Z{i}');
        b = B{i}(:,1)-rho.*sum(Delta{i},2);
        B{i} = repmat(b,[1,size(X,2)]);
    end
end
Error = sqrt(sum((Z{end}-t).^2)/N); 

figure;
scatter(1:numel(E),E);
xlabel('Iteration');
ylabel('E');

%% Plot the model
XAXIS = [-1:0.01:1];
Z{1} = XAXIS;
for i=1:numel(B)
    B{i} = repmat(B{i}(:,1),[1,numel(XAXIS)]);
end
for i=1:numel(Layers)-1
    A{i} = W{i}*Z{i}+B{i};
    h = H{i};
    Z{i+1} = h(A{i});
end
y = Z{numel(Layers)};

figure; 
hold on; 
scatter(X,t); 
plot(XAXIS,y);
legend('Data','Model');
xlabel('x');
ylabel('prediction');
title([num2str(Layers(2)) ' hidden units,' 'Erms:' num2str(Error)]);
hold off;

%% Repeat with 20 hidden units
clear;
rng(100);

%% Synthetic data
N = 100;
X=2*rand(1,N)-1;
t=sin(2*pi*X)+0.3*randn(1,N);

D = size(X,1);
Layers = [D,20,1]; % Number of units in each layers, input , hidden and output

figure;
scatter(X,t);
xlabel('Data');
ylabel('t');

%Activation Function
H{1} = @(x) tanh(x);
% H{2} = @(x) tanh(x);
H{2} = @(x) x;  
dH{1} = @(x) 1 - tanh(x).^2;
% dH{2} = @(x) 1 - tanh(x).^2;
dH{2} = @(x) 1;

%% Feed Forward Network
Z{1} = X;
W = cell(numel(Layers)-1,1);    % Weight Vector
B = cell(numel(Layers)-1,1);    % Bias term
Delta = cell(numel(Layers)-1,1);
rho = 0.002;

% Initial Guess Might need to be revised
for i=1:numel(W)
%     W{i} = 0.01+(0.1-0.01)*rand(Layers(i+1),Layers(i));
%     b = 0.01+(0.1-0.01)*rand(Layers(i+1),1);
    W{i} = rand(Layers(i+1),Layers(i));
    b = zeros(Layers(i+1),1);
    B{i} = repmat(b,[1,size(X,2)]);
end

Epoch = 8000;
for j=1:Epoch
    % Feed Forward
    for i=1:numel(Layers)-1
        A{i} = W{i}*Z{i}+B{i};
        h = H{i};
        Z{i+1} = h(A{i});
    end
    y = Z{numel(Layers)};
    
    % Back propagation
    Delta{end} = (y-t); % Output layer cost derivative
%     E(j) = sum(-t.*log(y)-(1-t).*log(1-y)); % Cost function
    E(j) = sum((y-t).^2); % Cost function
    for i=numel(Layers)-1:-1:1
        if i==1
            break;
        else
            dh = dH{i-1};
            Delta{i-1}  = W{i}'*Delta{i}.*dh(A{i-1});    % Delta(1)
        end
    end    
    for i=numel(Layers)-1:-1:1
        W{i} = W{i}-rho.*(Delta{i}*Z{i}');
        b = B{i}(:,1)-rho.*sum(Delta{i},2);
        B{i} = repmat(b,[1,size(X,2)]);
    end
end
Error = sqrt(sum((Z{end}-t).^2)/N); 

figure;
scatter(1:numel(E),E);
xlabel('Iteration');
ylabel('E');

%% Plot the model
XAXIS = [-1:0.01:1];
Z{1} = XAXIS;
for i=1:numel(B)
    B{i} = repmat(B{i}(:,1),[1,numel(XAXIS)]);
end
for i=1:numel(Layers)-1
    A{i} = W{i}*Z{i}+B{i};
    h = H{i};
    Z{i+1} = h(A{i});
end
y = Z{numel(Layers)};

figure; 
hold on; 
scatter(X,t); 
plot(XAXIS,y);
legend('Data','Model');
xlabel('x');
ylabel('prediction');
title([num2str(Layers(2)) ' hidden units,' 'Erms:' num2str(Error)]);
hold off;

close Figure 1 Figure 2 Figure 4 Figure 5