clear,clc;
close all;

%% Read
load carbig;

% Deal with the NaN data
index=find(isnan(Horsepower)==0);
Mean=mean(Horsepower(index));
Horsepower(isnan(Horsepower))=Mean;

% Normalize data
Weight_N = (Weight - min(Weight))/(max(Weight) - min(Weight));
n=numel(Weight_N);
%% Closed form
t=Horsepower;   % Target
t=(t-min(t))/(max(t)-min(t));

Mean_weight=mean(Weight_N);
Mean_t=mean(t);

Sxy=sum(t.*Weight_N)-(1/n)*sum(Weight_N)*sum(t);
Sxx=sum(Weight_N.^2)-(1/n)*(sum(Weight_N)^2);

b=Sxy/Sxx;
a=Mean_t-b*Mean_weight;

W=[b,a];
X=[Weight_N,ones(n,1)];
figure; 
hold on;
scatter(Weight_N,t);
plot(Weight_N,X*W');
hold off;