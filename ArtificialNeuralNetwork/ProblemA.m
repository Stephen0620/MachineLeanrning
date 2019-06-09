clear,clc;
close all;
 
%% Synthetic data
Class1 = [1,-1;-1,1]
Class2 = [1,1;-1,-1]
 
figure;
hold on;
scatter(Class1(:,1),Class1(:,2));
scatter(Class2(:,1),Class2(:,2));
xlabel('x');
ylabel('y');
hold off;
 
X = [Class1;Class2]';
% X = [-1,-1]'
t = [1,1,0,0];
% t = [1];
 
D = size(X,1);
Layers = [D,2,1]; % Number of units in each layers, input , hidden and output
 
%Activation Function
H{1} = @(x) ReLU(x,'Foward');
H{2} = @(x) 1./(1+exp(-x)); % Output Layer activation function
% dH{1} = @(x) exp(-x)./((exp(-x) + 1).^2);
dH{1} = @(x) ReLU(x,'Back');
dH{2} = @(x) exp(-x)./((exp(-x) + 1).^2);
 
%% Feed Forward Network
Z{1} = X;
W = cell(numel(Layers)-1,1);    % Weight Vector
B = cell(numel(Layers)-1,1);    % Bias term
Delta = cell(numel(Layers)-1,1);
rho = 0.5;

rng(15);
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
    E(j) = sum(-t.*log(y)-(1-t).*log(1-y)); % Cost function
%     E(j) = sum((y-t).^2); % Cost function
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
 
figure;
scatter(1:numel(E),E);
xlabel('Iteration');
ylabel('E');

%% plot
X = -1:0.1:1;
Y = -1:0.1:1;

clear Z;
for i=1:numel(B)
    B{i}(:,2:end) = [];
end

for x=1:numel(X)
    for y=1:numel(Y)
        Z{1} = [X(x);Y(y)];
        % Feed Forward
        for i=1:numel(Layers)-1
            A{i} = W{i}*Z{i}+B{i};
            h = H{i};
            Z{i+1} = h(A{i});
        end
        Result(x,y) = Z{end};
    end
end

%% Plot the decision surface
[X,Y] = meshgrid(-1:0.1:1,-1:0.1:1);
figure;
hold on;
grid on;
title('Decision Surface');
scatter3(Class1(:,1),Class1(:,2),ones(size(Class1,1),1),'x','LineWidth',2.0);
scatter3(Class2(:,1),Class2(:,2),zeros(size(Class2,2),1),'x','LineWidth',2.0);
s = surf(X,Y,Result);
view(3);
legend('Class1','Class2','Decision Surface','Location','north');
xlabel('x');
ylabel('y');
zlabel('prediction');

close Figure 1 Figure 2;