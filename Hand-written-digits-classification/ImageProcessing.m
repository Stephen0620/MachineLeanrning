clear,clc;
close all;

%% 
addpath('.\Handwritten data');
load TrainNet.mat;
Color = imread('1.jpg');
Color = imrotate(Color,90);

textBBoxes = Detection(Color);
% Color = insertShape(Color,'Rectangle',textBBoxes);
% figure; imshow(Color);

Digit = imresize(rgb2gray(imcrop(Color,textBBoxes(1,:))),[28,28]);
Digit = imcomplement(Digit);
Digit = double(Digit);
Digit = reshape(Digit,[28,28,1]);
figure; imshow(Digit,[]);

classify(trainedNet,Digit)
% 
% Gray = rgb2gray(Color);
% for i=1:size(textBBoxes,1)
%     I = imcrop(Gray,textBBoxes(i,:));
%     Gray = imresize(I,[28,28]);
%     classify(trainedNet,Gray);
% end
% classify(trainedNet);