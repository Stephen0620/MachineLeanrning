clear,clc;
close all;

%% Read Data
S = textread('Type.txt','%s');
D = xlsread('Data.xls');
Species = unique(S);
K = numel(Species);

class = ones(size(D,1),1);
for i=1:K
    idx = find(strcmp(S,Species{i})==1);
    class(idx) = i;
end

varNames={'feature1','feature2','feature3','feature4','feature5',...
    'feature6'}';
figure;
%gplotmatrix(x,y,group,clr,sym,siz,doleg,[],xnam,ynam)
gplotmatrix(D,[],class,['b' 'g' 'r'],'o',[1.5,1.5,1.5],'on',...
    'hist',varNames,varNames);

%% 1D View (One feature at a time)
Legend={'class1','class2','class3'};
for i=1:size(D,2)
    figure;
    hold on;
    for j=1:K
        scatter(D(class==j,i),class(class==j));
    end
    title(varNames{i});
    legend(Legend);
    hold off;
end