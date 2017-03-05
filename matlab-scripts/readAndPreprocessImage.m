% Read and pre-process images
% Copyright 2016 The MathWorks, Inc.
function Iout = readAndPreprocessImage(filename)

I = imread(filename);

% Blur Image with default standard deviation = 0.5
I = imgaussfilt(I);

% Change standard deviation of gaussian filter function to compare results
% I = imgaussfilt(I, 0.25);
% I = imgaussfilt(I, 0.5);
% I = imgaussfilt(I, 0.7);
% I = imgaussfilt(I, 0.1);
% I = imgaussfilt(I, 0.2);
% I = imgaussfilt(I, 0.3);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end

% Resize the image as required for the CNN.
Iout = imresize(I, [227 227]);

end
