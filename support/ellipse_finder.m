function varargout = ellipse_finder(varargin)
% FUNCTION ellipse_finder()
% FUNCTION rois = ellipse_finder()
% FUNCTION rois = ellipse_finder(image,diameter)
%
% This function uses Canny edge detection and the method of least-squares
% to find ellipses in noisy images. 
%
%OPTIONAL INPUTS:
%image: filename of tif image file (only the 1st frame will be read),
%       or a 2-D greyscale image matrix
%diameter: estimated diameter of ellipses to detect
% (if not all inputs are supplied, the user will be prompted to add them)
%
%OPTIONAL OUTPUT:
%rois: struct containing polygon vertices, centers, and parameters (e.g. 
%      radius, brightness, circularity) for each polygonal ROI


%% set required starting parameters
%load image
if nargin==1 %check if supplied image is a filename or an image matrix
    image = varargin{1};
    inputType = whos('image');
    if strcmp(inputType.class,'char')
        assert(exist(image,'file')==2,'cannot find input image file')
        [savePath, imageName] = fileparts(image);
        image = imread(image,1);
    else
        imageName = '';
        savePath = pwd;
        image = image(:,:,1);
    end
    image = varargin{1};

elseif nargin==0 %ask for image file
    [imageName, savePath] = uigetfile('*.tif','select a single-channel .tif file');
    assert(~strcmp(class(imageName),'double'),'No file selected');
    image = imread(fullfile(savePath,imageName));
end

%get estimate of ROI diameter
if nargin==2 %get diameter input
    diameter = varargin{2};
else %ask for diameter
    %get estimated diameter
    %%%%%%%%%todo: show image to help estimate
    diameter = 10;
    prompt = {'estimated ROI diameter (in pixels):'};
    dlgtitle = 'Input';
    dims = [1 40];
    definput = {num2str(diameter)};
    answers = inputdlg(prompt,dlgtitle,dims,definput);
    assert(~isempty(answers),'ROI diameter estimate is required');
    diameter = str2double(answers{1});
end

%for debugging ellipse detection:
plotEllipse = 0; %Ellipse # to plot fit details for (0 runs the script as normal)


%% load/pre-process image
[imageH, imageW] = size(image);
minRadius = round((1/3)*diameter); %minimum search radius for ellipses (in pixels)
maxRadius = round((2/3)*diameter); %maximum search radius for ellipses (in pixels)
blur = diameter/4; %size of gaussian blue, to remove small peaks
maxSize = round(1.25*diameter); %radius around potential ellipse center to analyze
numAngles = 1+2*maxSize; %number of angles in radial projection

procImage = imgaussfilt(image,blur); %process image with gaussian filter
procImage = imhmax(procImage,2); %suppress local maxima with height <2
procImage = imregionalmax(procImage); %find all local maxima
[procImage, numEllipses] = bwlabel(procImage); %uniquely label all 8-connected components

%get x,y centers of all connected components
x = repmat(1:imageW,[imageH 1]); %create subscripts for x and y positions
y = repmat((1:imageH)',[1 imageW]);
brightCenter = nan(numEllipses,2); %pre-allocate space for results
for e = 1:numEllipses %loop for every potential ellipse
    brightCenter(e,1) = round(mean(x(procImage==e),'all','omitnan'));
    brightCenter(e,2) = round(mean(y(procImage==e),'all','omitnan'));
end

%sort potential ellipses by max brightness
centerBrightnesses = image(sub2ind([imageH imageW],brightCenter(:,2),brightCenter(:,1)));
[~,sort_order] = sort(centerBrightnesses,'descend');
brightCenter = brightCenter(sort_order,:);


%% pre-compute x/y to radius/angle mapping stuff
xsq = repmat(-maxSize:maxSize,[numAngles 1]); %x position of each element relative to potential ellipse center
ysq = repmat((-maxSize:maxSize)',[1 numAngles]); %y position ""
asq = atan2(ysq,xsq); %angle ""
rsq = sqrt(xsq.^2 + ysq.^2); %radius ""

angIncr = pi/(maxSize+0.499); %angle between columns
asub = 1+maxSize+round(asq/angIncr); %angle subscripts of each element
rsub = 1+round(rsq); %radius subscripts of each element
usub = reshape(1:numel(xsq),size(xsq)); %unique subscript for each element
radialRegion = nan(numAngles, numAngles, numel(xsq)); %angle by radius [x by y]
inds = sub2ind(size(radialRegion),rsub(:),asub(:),usub(:)); %linear indices to place each element into abyr
newRsub = repmat((1:numAngles)',[1 numAngles]); %radial subscript of re-mapped image
newAsub =  xsq*angIncr; %angle subscript of re-mapped image


%% pre-compute array of ellipses
maxEdgePoints = 6*maxSize; %max number of detected edge elements to use fo fitting ellipses to edge elements
ellipseAngles = newAsub(1,:); %angles of each column

radiuses = minRadius:maxRadius; %different radiuses to try in ellipse fitting
xShifts = 0:(minRadius+1); %different ellipse center x position shifts to try
yShifts = 0:(minRadius+1); %different ellipse center y position shifts to try
eccentricities = 1:0.2:2.2; %different ellipse eccentricity amplitudes to try
eccAngles = 0:2*maxSize; %different eccentricity angle shifts to try

%get dimensions of 6-D array of all possible ellipses to try
dims = [length(radiuses) length(ellipseAngles) length(xShifts) length(yShifts) length(eccentricities) length(eccAngles)];

%create 6-D array of all ellipse parameters except for ecc_angle (I don't
%know the equation for an ellipse that parameterizes the eccentricity angle)
%dimension format: [radius, angle, x-shift, y-shift, eccentricity, ecc_angle]
r6 = repmat(permute(radiuses,[2 1]),[1 dims(2:6)]);
a6 = repmat(ellipseAngles,[dims(1) 1 dims(3:6)]);
x6 = repmat(permute(xShifts,[1 3 2]),[dims(1:2) 1 dims(4:6)]);
y6 = repmat(permute(yShifts,[1 3 4 2]),[dims(1:3) 1 dims(5:6)]);
e6 = repmat(permute(eccentricities,[1 3 4 5 2]),[dims(1:4) 1 dims(6)]);
e6 = e6.*r6; %scale eccentricities by radius

%get radius of every ellipse for all parameters except ecc_angle
%quadratic equation to solve for ellipse radius
a = ((cos(a6).^2)./(e6.^2)) + ((sin(a6).^2)./(r6.^2));
b = ((-2.*x6.*cos(a6))./(e6.^2)) - ((2*y6.*sin(a6))./(r6.^2));
c = ((x6.^2)./(e6.^2)) + ((y6.^2)./(r6.^2)) - 1;
d = (b.^2) - (4*a.*c);
d(d<=0) = nan;

ellipsesRad = (-b + sqrt(d))./(2*a); %calculate radius of all ellipses

%remove ellipses with any missing points (due to d<=0, when ellipse is off-center)
incompleteEllipses = repmat(sum(isnan(ellipsesRad),2)>0,[1 dims(2) 1 1 1 1]);
ellipsesRad(incompleteEllipses) = nan;

%remove too-large ellipses from array (e.g. large radiuses and eccentricities combine to big avg radiuses)
ellipsesAvgRad = repmat(mean(ellipsesRad,2),[1 dims(2) 1 1 1 1]);
ellipsesRad(ellipsesAvgRad>(maxRadius+0.5)) = nan;

%to calculate ellipses for different eccentricity angles, simply rotate
%every ellipse for each ecc_angle (rotation = circshift in the angle dimension)
for i = 2:length(eccAngles) 
    ellipsesRad(:,:,:,:,:,i) = circshift(ellipsesRad(:,:,:,:,:,1),[0 eccAngles(i) 0 0 0 0]);
end
ellipsesRad = 1+ellipsesRad; %add 1 so that radius0 = row1


%% find ellipses by looping for every maxima (ie potential ellipse)
%pre-allocate space for results

% parameters = [radius, brightness, eccentricity, distanceToEdge, fractionEnclosed, uncertainy]
% rois.vertices = {numPts, 2} <- (cell array of [x,y] values)
% rois.centers = (numRois, 2) <- [x,y] values of the roi actual center (not bright center)
rois.parameters = nan(numEllipses,6);
rois.vertices = cell(numEllipses,1);
rois.center = nan(numEllipses,2);

roiRadius = nan(numEllipses, numAngles);
roiVertices = nan(numAngles, 2);

%parameters
uncertainty = nan(numEllipses,1);
diameters = nan(numEllipses,1);
circularity = nan(numEllipses,1);
brightness = nan(numEllipses,1);
fractionEnclosed = nan(numEllipses,1);
distanceToEdge = nan(numEllipses,1);

if plotEllipse>0
    maximaToAnalyze = plotEllipse;
else
    maximaToAnalyze = 1:numEllipses;
end
waitbarHandle = waitbar(0,['Processing ' num2str(numEllipses) ' potential ellipses']);
for e = maximaToAnalyze %loop for every maxima
    waitbarHandle = waitbar(e/numEllipses,waitbarHandle);

    %get square region around maxima
    squareRegion = nan(numAngles); %pre-allocate space
    xlims = [1 numAngles]; %current limits of square region
    ylims = [1 numAngles];
    
    xrange = [brightCenter(e,1)-maxSize brightCenter(e,1)+maxSize];
    yrange = [brightCenter(e,2)-maxSize brightCenter(e,2)+maxSize];
    
    %prevent square region from extending outside of input image bounds
    xlims(1) = xlims(1) - min([0 xrange(1)-1]); 
    xrange(1) = max([1 xrange(1)]);
    xlims(2) = xlims(2) + min([0 imageW-xrange(2)]);
    xrange(2) = min([imageW xrange(2)]);
    ylims(1) = ylims(1) - min([0 yrange(1)-1]);
    yrange(1) = max([1 yrange(1)]);
    ylims(2) = ylims(2) + min([0 imageH-yrange(2)]);
    yrange(2) = min([imageH yrange(2)]);
    
    %get square region (only including pixels within image bounds)
    squareRegion(ylims(1):ylims(2),xlims(1):xlims(2)) = image(yrange(1):yrange(2),xrange(1):xrange(2));

    %tranform square region into angle/radius projection
    radialRegion = nan(numAngles,numAngles,numel(xsq)); %angle by radius [x by y]
    radialRegion(inds) = squareRegion(:);
    radialRegion = round(mean(radialRegion,3,'omitnan')); %abyr will still have some nans at small radiuses, need to fill them
    radialRegion = fillmissing(radialRegion,'nearest',2);
    radialRegion(isnan(radialRegion)) = 0;
    squareRegion(isnan(squareRegion)) = 0;
    maximumRadius = round(sqrt(2)*maxSize) + 1; %this is furthest from the center that any point can be
    blankRegion = zeros(size(radialRegion));
    
    %use canny edge detection to find boundaries of potential ellipse
    edgeImage = edge([radialRegion(:,end) radialRegion radialRegion(:,1)],'canny'); %wrap for edge detection
    edgeImage = edgeImage(:,2:end-1); %cut out wrapped edges
    
    %only use edges that go from bright->dark
    edgeInds = find(edgeImage); %indices of edge points
    [edgeRowSub, edgeColSub] = ind2sub(size(edgeImage),edgeInds); %subscripts
    innerRowSub = max([ones(size(edgeRowSub)) edgeRowSub-1],[],2,'omitnan'); %1 is the innermost row
    outerRowSub = min([maximumRadius*ones(size(edgeRowSub)) edgeRowSub+1],[],2,'omitnan'); %maximum_radius is the outermost row
    innerInds = sub2ind(size(edgeImage),innerRowSub,edgeColSub);
    outerInds = sub2ind(size(edgeImage),outerRowSub,edgeColSub);
    innerBrightness = radialRegion(innerInds); %get brightness value on inner side of edge
    outerBrightness = radialRegion(outerInds); %get brightness value on outer side of edge
    innerOuter = innerBrightness-outerBrightness;
    innerNotBrighter = innerOuter<=0; %edge inds that aren't going from bright->dark
    edgeIndsNotBrighter = edgeInds(innerNotBrighter);
    edgeImage(edgeIndsNotBrighter) = 0; %erase edge inds that aren't going from bright->dark
    
    %fit ellipse to edge points
    [edgecols,edgerows] = find(edgeImage',maxEdgePoints); %get indices a bunch of nearest edge points closest to the ellipse center
    curellipses = ellipsesRad(:,edgecols,:,:,:,:); %re-order ellipse radiuses to match the columns of each edge point
    errs = abs(curellipses - repmat(edgerows',[dims(1) 1 dims(3:6)])); %get difference between ellipses and edgepoint ("errors") of the same column (ie angle)
    
    %for columns (ie angles) with multiple edge points, use the edge point that's closest to the ellipse
    colInds = nan(size(radialRegion));
    colInds(sub2ind(size(colInds),edgerows,edgecols)) = 1:length(edgecols);
    cols_withmultiple = find(sum(colInds>0)>1); %columns that have multiple edge points
    for col = 1:length(cols_withmultiple)
        repeat_inds = colInds(:,cols_withmultiple(col)); %indices of edge points of the same column
        repeat_inds(isnan(repeat_inds)) = [];
        errs(:,repeat_inds(1),:,:,:,:) = min(errs(:,repeat_inds,:,:,:,:),[],2); %replace the first instance with the closest instance
    end
    colsUnique = min(colInds,[],1,'omitnan'); %only use the first instance (now closest instance) of edge point
    colsUnique(isnan(colsUnique)) = [];
    numColsUnique = length(colsUnique); %number of columns with at least 1 edge point
    errs = errs(:,colsUnique,:,:,:,:); %only use columns with at least 1 edge point, and only the closest one of each
    
    %reduce affect of outliers in errs
    errStd = repmat(std(errs,[],2),[1 numColsUnique 1 1 1 1]); %get standard deviation of ellipse-edgepoint differences
    errs(errs>3*errStd) = nan; %remove outliers (3*S.D.)
    maxErrs = repmat(max(errs,[],2,'omitnan'),[1 numColsUnique 1 1 1 1]);
    errs(isnan(errs)) = 1 + maxErrs(isnan(errs)); %change remove outliers to be just barely the biggest difference
    
    %calculate sum of squared errors
    sse = sqrt(sum(errs.^2,2)); %sumresults.uncertainty(191) of squared errros for each ellipse
    [sumsqerror, minInd] = min(sse,[],'all','linear'); %find the ellipse with the smallest sse (ie best fit to edge points)
    %higher sse = higher uncertainty of whether an ellipse is a good fit
    uncertainty(e) = sumsqerror;
    fractionEnclosed(e) = numColsUnique/numAngles; %quantify the amount of angles missing edge points entirely
    %get indices of best fitting ellipse
    [rInd, ~, pxInd, pyInd, emInd, enInd] = ind2sub(size(sse),minInd);
    
    %get radius (as a function of angle) of best fitting ellipse
    roiRadius(e,:) = ellipsesRad(rInd,:,pxInd,pyInd,emInd,enInd) - 1; %subtract 1, since row1 = radius0
    diameters(e) = 2*mean(roiRadius(e,:),'omitnan');
    
    %get polygon roi
    roiVertices(:,1) = (roiRadius(e,:).*cos(ellipseAngles) + brightCenter(e,1))';
    roiVertices(:,2) = (roiRadius(e,:).*sin(ellipseAngles) + brightCenter(e,2))';
    rois.vertices{e} = roiVertices;

    %get actual ellipse center
    rois.center(e,1) = mean(roiVertices(:,1),'omitnan');
    rois.center(e,2) = mean(roiVertices(:,2),'omitnan');
    
    %get distance to edge
    xEdgeDist = min([rois.center(e,1) imageW-rois.center(e,1)]);
    yEdgeDist = min([rois.center(e,2) imageH-rois.center(e,2)]);
    distanceToEdge(e) = min([xEdgeDist yEdgeDist]);
    
    %get roi circularity
    polyin = polyshape(roiVertices(:,1),roiVertices(:,2));
    roiArea = area(polyin);
    roiPerimeter = perimeter(polyin);
    circularity(e) = roiPerimeter^2/(4*pi*roiArea);
    
    %get roi mask
    mask = poly2mask(roiVertices(:,1),roiVertices(:,2),imageH,imageW);
    brightness(e) = mean(image(mask),'omitnan');

    %if desired, plot ellipse detection for the current ellipse to figure
    if e == plotEllipse
        figure('Position',[100 100 1500 300])
        subplot(1,5,1)
        image(squareRegion);
        xlabel('x')
        ylabel('y')
        title('possible ellipse')

        subplot(1,5,2)
        image(radialRegion);
        xlabel('angle')
        ylabel('radius')
        title('radial projection')

        subplot(1,5,3)
        image(radialRegion + edgeImage*100);
        title('"canny" edge detection')

        subplot(1,5,4)
        ellipseImage = zeros(size(radialRegion));
        ellipseImage(sub2ind(size(ellipseImage),round(roiRadius(e,:)+1),1:numAngles)) = 1;
        image(radialRegion + ellipseImage*100);
        title('ellipse fit')

        subplot(1,5,5)
        image(image)
        patch(roiVertices([1:end 1],1),roiVertices([1:end 1],2),'k')
        xlim([brightCenter(e,1)-25 brightCenter(e,1)+25])
        ylim([brightCenter(e,2)-25 brightCenter(e,2)+25])
        title('ellipse roi')
    end
end
close(waitbarHandle);
if plotEllipse>0
    return
end

rois.parameters = [diameters, circularity, brightness, distanceToEdge, fractionEnclosed, uncertainty];
rois.parameterInfo = '[diameters, circularity, brightness, distanceToEdge, fractionEnclosed, uncertainty]';

if nargout==0
    fullSaveName = fullfile(savePath,[imageName(1:end-4) 'ellipse_finder.mat']);
    save(fullSaveName,'rois');
    msgbox(['results saved as: ' fullSaveName]);
else
    varargout{1} = rois;
end
