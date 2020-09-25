function out = read_tiffstate(tiff,statename)
%FUNCTION out = read_tiffstate(tiff,name)
%
%reads values stored by scanimage as "state" values in the 
%"ImageDescription" tag of Tiff files. 
%
%INPUTS
%tiff: matlab tiff object
%name: name of state field
%
%OUTPUTS
%out: value for the selected "state" field
%
%e.g.
%tiff = Tiff('C:/image.tif','r');
%fps = read_tiffstate(tiff,'framerate');

out = nan;

%get entire "ImageDescription" string using the matlab Tiff library
try
    descr = getTag(tiff,'ImageDescription');
catch
    disp('Could not get "ImageDescription" tag in Tiff file');
    return
end

%find the desired state
name_ind = strfind(descr,statename);
if isempty(name_ind)
    disp(['Could not find "' statename '" in Tiff file ImageDescription tag'])
    return
end

%find where the next state starts (i.e. the end of the desired state)
end_ind = strfind(descr,'state');
end_ind = end_ind(find(end_ind>name_ind,1));

%get the string of the state field with 2 possible delimiters
state_str = descr(name_ind:end_ind-1);
eq_ind = strfind(state_str,'=');
ap_ind = strfind(state_str,'''');

%return the state value as either a string or number
if isempty(ap_ind)
    out = str2num(state_str(eq_ind+1:end)); %equal sign begins number
else
    out = state_str(ap_ind(1)+1:ap_ind(2)-1); %apostrophes surround string
end

end



