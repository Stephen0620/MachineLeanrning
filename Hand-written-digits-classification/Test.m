clear,clc;
close all;
addpath('.\Handwritten data');

Color = imread('1.jpg');
Color = imrotate(Color,90);
figure; imshow(Color);
% Color = imcrop(Color,[145,268,633,200]);
gray = rgb2gray(Color);

background = imopen(gray,strel('disk',11));
gray = gray - background;
figure; imshow(gray);

gray = imadjust(gray);
Thresh = graythresh(gray);
BW = imbinarize(gray,Thresh);
% BW = imcomplement(BW);

figure; imshow(BW);

CC = bwconncomp(BW);
CCStats = regionprops(CC,'Area','BoundingBox','FilledImage','EulerNumber');

idx = find([CCStats.Area]<100 | [CCStats.Area]>500);
CCStats(idx)=[];
BBoxes = vertcat(CCStats.BoundingBox);

Color = insertShape(Color,'Rectangle',BBoxes);
figure; imshow(Color);