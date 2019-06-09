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

varNames={'Feature1','Feature2','Feature3','Feature4','Feature5',...
    'Feature6'}';
figure;
%gplotmatrix(x,y,group,clr,sym,siz,doleg,[],xnam,ynam)
gplotmatrix(D,[],class,['b' 'g' 'r'],'o',[1.5,1.5,1.5],'on',...
    'none',varNames,varNames);
title('GplotMatrix');

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
    xlabel('Samples');
    ylabel('Class');
    hold off;
end