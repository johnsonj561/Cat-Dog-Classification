%% CAP 4630 - Intro to AI - FAU - Dr. Marques - Fall 2016 
% Justin Johnson, Sam Rosenfield, Nick
% Directions:
% Run Parts 1 - 4
% Part 5 Requires User to Open Classification Learner App to Train Models
% Run Parts 6 - 7 on 1st model, record results
% Edit model in Part 5
% Run Parts 6 - 7 on 2nd model, record results
% Compare Results

%% Part 1: Download, load and inspect Pre-trained Convolutional Neural Network (CNN)
% PreTrained CNN "AlexNet" is being used for feature extraction

%% 1.1: Location of pre-trained "AlexNet"
% Define location of AlexNet
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';

% Specify where AlexNet CNN will be stores
cnnFolder = 'networks';
cnnMatFile = 'imagenet-caffe-alex.mat'; 
cnnFullMatFile = fullfile(cnnFolder, cnnMatFile);

% Only download CNN once
if ~exist(cnnFullMatFile, 'file')
    disp('Downloading pre-trained CNN model...');     
    websave(cnnFullMatFile, cnnURL);
else
    disp('CNN was previously defined.');
end

%% 1.2: Load Pre-trained CNN
% Load MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet('imagenet-caffe-alex.mat');

%% Part 2: Set up image data
%% 2.1: Load simplified dataset and build image store
dataFolder = 'data/PetImages';
categories = {'cat', 'dog'};
imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);
tbl = countEachLabel(imds);

% Use the smallest overlap set
% (useful when the two classes have different number of elements)
minSetCount = min(tbl{:,2});

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% 2.2: Pre-process Images For CNN
%Convnet requires 227x227 pixel RGB Images
%Image re-sizing and Pre-processing techniques such as blurring, sharpenning, contrast
%enhancement, etc are defined in the readAndPreprocessImage.m script
%
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

%% 2.3: Divide data into training, validation, and testing sets
% Partition data set into training, cross-validation, and testing
% Recommended is 60% / 20% / 20%
% Since we will be using classification learner app to generate models, we
% partition data into 80/20, leaving 20% for testing. The 80% will then be
% used by classification learner app, which will apply 75% /  25% partition into
% training set and validation set. This will generate the desired
% partitions of:
% 60% training
% 20% cross-validating
% 20% testing
[trainingAndValidationSet, testSet] =  splitEachLabel(imds, 0.8, 'randomize');

%% Part 3: Feature Extraction 
% Extract training features using pretrained CNN

%% 3.1: Use features from one of the deeper layers
% Selecting which of the deep layers to choose is a
% design choice, but typically starting with the layer right before the
% classification layer is a good place to start. In |convnet|, the this
% layer is named 'fc7'. Let's extract training features using that layer.
% fc8 is an alternative that could be used to produce different features

featureLayer = 'fc7';
%featureLayer = 'fc8';

trainingFeaturesFolder = './';
trainingFeaturesFile = 'trainingFeatures.mat'; 
trainingFeaturesFullMatFile = fullfile(trainingFeaturesFolder, trainingFeaturesFile);

% Check that the code is only downloaded once
if ~exist(trainingFeaturesFullMatFile, 'file')
    disp('Building training features... This will take a while...');     
    trainingFeatures = activations(convnet, trainingAndValidationSet, featureLayer, ...
      'MiniBatchSize', 32, 'OutputAs', 'columns');
    save(trainingFeaturesFullMatFile, 'trainingFeatures');
else
    disp('Loading training features already defined');
    load trainingFeatures.mat
end
%% Part 4: Generate A Table For Classification Learner App
% The Classification Learner App requires table format to read
% predictors and results
trainingLabels = trainingAndValidationSet.Labels;
trainingTable = table(trainingFeatures', trainingLabels);

%% Part 5: Run Classification Learner App
% Open Classification Learner App
% Select 'trainingTable' from workspace as data set
% Train All Discriminant Analysis Models
% Once training is complete:
% Select Linear Discriminant and export "Compact Model" to workspace 
%   as trainedClassifier1
% Select Quadratic Discriminant and export "Compact Mode" to workspace
%   as trainedClassifier 2

% Note - both models will be used to compare results.

%% Part 6: Utilize Trained Model From Classification Learner App
% Get training labels from the trainingSet

% 1st Iteration Use trainedClassifier1 and record results
classifier = trainedClassifier1.ClassificationDiscriminant;

% 2nd Iteration Use trainedClassifier2 and record results
% classifier = trainedClassifier2.ClassificationDiscriminant;

%% Part 7: Evaluate classifier

%% 7.1: Extract features from images in the test set
testFeaturesFolder = './';
testFeaturesFile = 'testFeatures.mat'; 
testFeaturesFullMatFile = fullfile(testFeaturesFolder, testFeaturesFile);

% Check that the code is only downloaded once
if ~exist(testFeaturesFullMatFile, 'file')
    disp('Extracting features... This will take a while...');     
    % Extract test features using the CNN
    testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);
    % Save features for future use
    save(testFeaturesFullMatFile, 'testFeatures');
else
    disp('Loading previously defined testFeatures.mat');
    load testFeatures.mat
end

%% 7.2: Test classifier's prediction accuracy and produce confusion matrix
% Pass CNN image features from test set to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2)) 

%% 7.3: Test it on an unseen image
newImage = 'data/doge.jpg';
img = readAndPreprocessImage(newImage);
imageFeatures = activations(convnet, img, featureLayer);
label = predict(classifier, imageFeatures);

% Display test image and assigned label
figure, imshow(img), title(char(label)); 

%% References
% [1] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image
% database." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE
% Conference on. IEEE, 2009.
%
% [2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
% classification with deep convolutional neural networks." Advances in
% neural information processing systems. 2012.
%
% [3] Vedaldi, Andrea, and Karel Lenc. "MatConvNet-convolutional neural
% networks for MATLAB." arXiv preprint arXiv:1412.4564 (2014).
%
% [4] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding
% convolutional networks." Computer Vision-ECCV 2014. Springer
% International Publishing, 2014. 818-833.
%
% [5] Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature
% for generic visual recognition." arXiv preprint arXiv:1310.1531 (2013).
