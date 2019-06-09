clear,clc;
close all;

cam = webcam(2);
cam.Resolution = cam.AvailableResolution{4};
Rect = [161,235,311,140];
figure(1); 
while(true) 
    Color = snapshot(cam);
    Color = insertShape(Color,'Rectangle',[161,180,312,247]);
    ROI = imcrop(Color,Rect);
    TextBBoxes = Detection(ROI);
    if isempty(TextBBoxes)
        imshow(Color);
        continue;
    end
    TextBBoxes(:,1) = TextBBoxes(:,1)+Rect(1)-1;
    TextBBoxes(:,2) = TextBBoxes(:,2)+Rect(2)-1;
    Color = insertShape(Color,'Rectangle',TextBBoxes,'Color',[0,0,0]);
    imshow(Color);
end