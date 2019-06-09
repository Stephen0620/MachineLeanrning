%% Generate testing set
N_test=100;
x_test=rand(N_test,1);
Epsolon_test=randn(N_test,1)*0.3;
t_test=sin(2*pi*x_test)+Epsolon_test;

%% Generate Phi for testing set
Phi_test=cell(Total_M,1);
for i=0:numel(Phi_test)-1
    M=i;
    Phi_test{i+1}=Getting_Phi(x_test,M);
end
