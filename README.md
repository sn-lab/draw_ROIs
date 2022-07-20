# draw_ROIs
 Matlab app for manually drawing ROIs (regions of interest) and extracting traces of mean pixel intensity values within those ROIs from scanimage .tif files

DEPENDENCIES: 
* Image Processing Toolbox
* Statistics and Machine Learning Toolbox

##
ROIs are commonly used to identify spatial subsets of an image series in order to calculate how the mean intensity of all pixels within that region changes throughtout the image series. This app simplifies the process for identifying and drawing these ROIs.

The app can either be run directly or called as a function from other Matlab scripts. When ran directly or called as a function with no inputs (e.g. "draw_ROIs()"), the "draw_ROIs" app will open with only the "Load Image Series" button visible, prompting the user to select an image file located in their file system for drawing ROIs. Alternatively, an image file can be passed into the app as an input, bypassing the need for the user to select a file after the app has loaded. An image file can be passed into the function as either an image matrix already loaded in the matlab workspace (e.g. "draw_ROIs(image_series)") or as a file directory pointing to a .tif image file (e.g. "draw_ROIs('C:/image.tif')").

When loading an image file from a directory, the app will attempt to read the framerate that the image series was acquired; while not necessary for the app to function, the framerate can be useful in plotting the time-course of the ROI intensity changes afterwards. If the app is unable to find the framerate information from the Tiff file metadata, a dialog box will open asking the user to input the framerate directly. Alternatively, when calling the app as a function, the image framerate can be passed into the app as the second input (e.g. "draw_ROIs('C:/image.tif',30)").

After an image series has been loaded, a mean projection of the image will be displayed. Additionally, a dropdown box above the image will let the user select between a mean projection of the image or a "mean + variance" projection, with the intensity variance of the image displayed in red. This option may be useful in finding ROIs with highly fluctuating intensity values.

To begin drawing ROIs using the mean image projection (or mean + variance projection) as a guide, clicking the "add ROI" button will open a separate window prompting the user to draw a polygon around the desired ROI. Vertices of the polygon can be set by single clicks around the ROI. After placing a number of vertices, the ROI polygon can be enclosed/finalized by double-clicking the starting vertex. 

After all desired ROIs have been drawn, clicking the "Save and Exit" button will save two matlab .mat files in the same directory as the image file. The first .mat file (filename begining with "rois...") contains the ROI polygon vertex coordinates as well as binary image masks of all drawn ROIs -- this file can be re-loaded back into the app for later visualization/editing/etc. The 2nd .mat file (filename begining with "traces...") contains the mean intensity value traces of all drawn ROIs, using the binary image masks described by the ROIs. These vectors can be plotted to visualize how the mean intensity value of each ROI changes across the image series (i.e. changes with time). Additionally, image files (either .png, .svg, or both) can be optionally saved which display the ROIs drawn on top of the mean projection.
