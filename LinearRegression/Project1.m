clear,clc;
close all;

%% Read
load carbig;

% Deal with the NaN data
index=find(isnan(Horsepower)==0);
Mean=mean(Horsepower(index));
Horsepower(isnan(Horsepower))=Mean;

% Normalize data
Mean=mean(Weight(:));
Std=std(Weight(:));
Weight_N = (Weight - Mean)/Std;

%% Closed form
X=[Weight_N ones(numel(Weight_N),1)];   % Designed Matrix
t=Horsepower;   % Target

w_close=inv(transpose(X)*X)*transpose(X)*t;
Closed_Cost=sum((X*w_close-t).^2);

figure;
plot(Weight,X*w_close,'r');
hold on;
scatter(Weight,Horsepower,'b');
title('Matlab "Carbig" dataset');
legend('Closed Form');
xlabel('Weight');
ylabel('Horsepower');
hold off;

%% Iterative Solution
initial_guess=[0;0];
rho=0.00001;

w_iterative=initial_guess;

terminal=false;
times=0;
Max_iteration=100000;
while(~terminal)
    times=times+1;
    w_old=w_iterative;
    gradient=transpose(2.*transpose(w_old)*transpose(X)*X-2.*transpose(t)*X);
    
    % Terminal Condition
    if all(abs(gradient)<1e-6)||(times==Max_iteration)
        terminal=true;
    end 
    w_iterative=w_old-rho*gradient;        
end

Closed_Iterative=sum((X*w_iterative-t).^2);

figure; 
plot(Weight,X*w_iterative,'r');
hold on;
scatter(Weight,Horsepower,'b');
title('Matlab "Carbig" dataset');
legend('Iterative Method');
xlabel('Weight');
ylabel('Horsepower');
hold off;

%% Stochastic
w_Stochastic=[0;0];
rho=1/(2*numel(t));
Epoch = 50;

for j=1:Epoch
    for i=1:numel(t)
    %     if i==1
    %         sigma = 1;
    %     else
    %         sigma = 1./(1+exp(-X(i-1,:)*w));
    %     end
        w_Stochastic = w_Stochastic-((rho).*(X(i,:)*w_Stochastic-t(i)).*X(i,:))';
    end
end

Closed_Stochastic=sum((X*w_Stochastic-t).^2);

figure; 
plot(Weight,X*w_Stochastic,'r');
hold on;
scatter(Weight,Horsepower,'b');
title('Matlab "Carbig" dataset');
legend('Stochastic Method');
xlabel('Weight');

%% Cost function
Closed_Cost
Closed_Iterative
Closed_Stochastic
ylabel('Horsepower');
hold off;