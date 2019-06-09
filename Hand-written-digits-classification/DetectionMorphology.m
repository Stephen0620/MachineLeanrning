function textBBoxes = DetectionMorphology(ColorImage)
I = rgb2gray(ColorImage);
% I = histeq(I);

background = imopen(I,strel('disk',15));
I = I - background;

I = imadjust(I);

bw = imbinarize(I);
bw = bwareaopen(bw, 50);
% imshow(bw)

cc = bwconncomp(bw, 4);
ccStats = regionprops(cc, 'BoundingBox','Centroid','Area');

idx = [ccStats.Area]<100;
ccStats(idx) = [];

bboxes = vertcat(ccStats.BoundingBox);

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expand the bounding boxes by a small amount.
expansionAmount = 0.1;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;

% Clip the bounding boxes to be within the image bounds
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

% Show the expanded bounding boxes
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

%%
% Compute the overlap ratio
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1); 
overlapRatio(1:n+1:n^2) = 0;

% Create the graph
g = graph(overlapRatio);

% Find the connected text regions within the graph
componentIndices = conncomp(g);

%%
% Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

% Compose the merged bounding boxes using the [x y width height] format.
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
idx = find(ymin<5|ymax>size(I,1)-20);
textBBoxes(idx,:) = [];

%%
% Remove bounding boxes that only contain one text region
numRegionsInGroup = histcounts(componentIndices);
% textBBoxes(numRegionsInGroup == 1, :) = [];


end

