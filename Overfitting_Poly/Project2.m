clear,clc;
close all;

rng(103);
%% Generate the training set N=10
N=10;
x_original=rand(N,1);
Epsilon=randn(N,1)*0.3;
t=sin(2*pi*x_original)+Epsilon;

% figure;
% hold on;
% scatter(x_original,t);
% plot((0:0.01:1),sin(2*pi*(0:0.01:1)));
% hold off;

% Normalize the data
Mean=mean(x_original);
Std=std(x_original);
x=(x_original-Mean)/Std;

%% Generate Phi Matrix
Total_M=10;
Phi=cell(Total_M,1);
for i=0:numel(Phi)-1
    M=i;
    Phi{i+1}=Polynomial_Phi(x,M);
end

%% Closed Form
W=cell(Total_M,1);
% Lambda=0.5;
for i=1:numel(W)
%     Regularization
%     W{i}=inv(Phi{i}'*Phi{i}+Lambda.*eye(size(Phi{i},2)))*Phi{i}'*t; 
    W{i}=pinv(Phi{i})*t;
end

%% Errors
Errors=zeros(Total_M,1);
for i=1:numel(Errors)
    Errors(i)=sqrt(sum((t-Phi{i}*W{i}).^2)/N);
end

%% Generate testing set
N_test=100;
x_test_original=rand(N_test,1);
Epsolon_test=randn(N_test,1)*0.3;
t_test=sin(2*pi*x_test_original)+Epsolon_test;

% Normalize the testing data
x_test=(x_test_original-Mean)/Std;

%% Generate Phi for testing set
Phi_test=cell(Total_M,1);
for i=0:numel(Phi_test)-1
    M=i;
    Phi_test{i+1}=Polynomial_Phi(x_test,M);
end

%% Errors of testing set
Errors_test=zeros(Total_M,1);
for i=1:numel(Errors)
    Errors_test(i)=sqrt(sum((t_test-Phi_test{i}*W{i}).^2)/N_test);
end

%% Plotting of Errors
figure;
hold on;
plot((0:Total_M-1),round(Errors,2),'-o');
plot((0:Total_M-1),round(Errors_test,2),'-or');
legend('Training Erms','Test Erms');
title('Number of training data: 10');
xlabel('M');
ylabel('Erms');
hold off;

%% Plotting Model
x_axis=[min(x):0.01:max(x)];
figure;
hold on;
scatter(x,t);
plot(x_axis,Polynomial_Phi(x_axis,1)*W{2});
hold off;
figure;
hold on;
scatter(x,t);
plot(x_axis,Polynomial_Phi(x_axis,3)*W{4});
scatter(x_test,t_test);
figure;
hold on;
scatter(x,t);
plot(x_axis,Polynomial_Phi(x_axis,8)*W{9});
hold off;
figure;
hold on;
scatter(x,t);
plot(x_axis,Polynomial_Phi(x_axis,9)*W{10});
hold off;

%% Generate the training set N=100
N=100;
x_original=rand(N,1);
Epsilon=randn(N,1)*0.3;
t=sin(2*pi*x_original)+Epsilon;

% figure;
% hold on;
% scatter(x_original,t);
% plot((0:0.01:1),sin(2*pi*(0:0.01:1)));
% hold off;

% Normalize the data
Mean=mean(x_original);
Std=std(x_original);
x=(x_original-Mean)/Std;

%% Generate Phi Matrix
Total_M=10;
Phi=cell(Total_M,1);
for i=0:numel(Phi)-1
    M=i;
    Phi{i+1}=Polynomial_Phi(x,M);
end

%% Closed Form
W=cell(Total_M,1);
% Lambda=0.5;
for i=1:numel(W)
%     Regularization
%     W{i}=inv(Phi{i}'*Phi{i}+Lambda.*eye(size(Phi{i},2)))*Phi{i}'*t; 
    W{i}=pinv(Phi{i})*t;
end

%% Errors
Errors=zeros(Total_M,1);
for i=1:numel(Errors)
    Errors(i)=sqrt(sum((t-Phi{i}*W{i}).^2)/N);
end

%% Generate testing set
N_test=100;
x_test_original=rand(N_test,1);
Epsolon_test=randn(N_test,1)*0.3;
t_test=sin(2*pi*x_test_original)+Epsolon_test;

% Normalize the testing data
x_test=(x_test_original-Mean)/Std;

%% Generate Phi for testing set
Phi_test=cell(Total_M,1);
for i=0:numel(Phi_test)-1
    M=i;
    Phi_test{i+1}=Polynomial_Phi(x_test,M);
end

%% Errors of testing set
Errors_test=zeros(Total_M,1);
for i=1:numel(Errors)
    Errors_test(i)=sqrt(sum((t_test-Phi_test{i}*W{i}).^2)/N_test);
end

%% Plotting of Errors
figure;
hold on;
plot((0:Total_M-1),round(Errors,2),'-o');
plot((0:Total_M-1),round(Errors_test,2),'-or');
legend('Training Erms','Test Erms');
title('Number of Training data: 100');
xlabel('M');
ylabel('Erms');
hold off;
