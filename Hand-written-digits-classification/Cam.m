% if ~isempty(obj)
%     release(obj);
%     clear obj;
% end
close all;
clear;
%Get the input device using image acquisition toolbox,resolution = 640x480 to improve performance
obj =imaq.VideoDevice('winvideo', 1, 'YUY2_640x640');
set(obj,'ReturnedColorSpace', 'rgb');
set(obj,'ReturnedDataType','uint8');
figure('tag','webcam');

for i=1:100
    frame = step(obj);
%     bbox = DetectionMorphology(frame);
%     frame = insertShape(frame,'Rectangle',bbox);
% %     frame = insertText(frame,[1,1],'5');
    imshow(frame);
end
release(obj);
    