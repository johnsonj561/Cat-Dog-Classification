function net = helperImportMatConvNet(matfile)
% Imports MatConvNet network into a SeriesNetwork object from the Neural
% Network Toolbox. This function is provided for educational purposes only
% and does not support all models from MatConvNet.
%
% This helper function uses internal functions, which may change in a
% future release. This function requires the Neural Network Toolbox.

% References
% ----------
% Vedaldi, Andrea, and Karel Lenc. "MatConvNet-convolutional neural
% networks for MATLAB." arXiv preprint arXiv:1412.4564 (2014).

isInstalled = isdir(fullfile(matlabroot, 'toolbox', 'nnet'));
if ~isInstalled
    error('This function requires Neural Network Toolbox.');
end

matconvnet = load(matfile);

numLayers = numel(matconvnet.layers);

% Create image input layer and set average image for zero centering.
if isfield(matconvnet, 'meta')
    % newer versions of MatConvNet
    layers{1} = createInputImageLayer(matconvnet.meta.normalization);
else
    % older versions of MatConvNet (beta16)
    layers{1} = createInputImageLayer(matconvnet.normalization);
end

outputSize = layers{1}.InputSize;

for i = 1:numLayers
    
    l = matconvnet.layers{i};
    
    switch l.type
        case 'conv'            
            if isConv2DLayer(l, outputSize)                
                layers{end+1} = createConv2DLayer(l, outputSize);                                
            else
                layers{end+1} = createFullyConnectedLayer(l);
            end 
            
            % set default learning rates and use default initializer            
            layers{end}.Weights.L2Factor = 1;
            layers{end}.Weights.LearnRateFactor = 1;           
            layers{end}.Bias.L2Factor = 1;
            layers{end}.Bias.LearnRateFactor = 1;
                        
        case 'relu'                                                            
            layers{end+1} = createReLULayer(l);
            
        case 'pool'
            layers{end+1} = createPoolLayer(l);
            
        case 'dropout'
            layers{end+1} = createDropoutLayer(l);
            
        case {'normalize','lrn'}
            layers{end+1} = createLocalMapNorm2DLayer(l);
            
        case 'softmax'                                                                                       
            
            % output size matches the current end of the layers.
            numNeurons = layers{end}.NumNeurons;
            layers{end+1} = nnet.internal.cnn.layer.Softmax(l.name);
            
            layers{end+1} = nnet.internal.cnn.layer.CrossEntropy('classificationLayer', numNeurons);
            layers{end}.ClassNames = matconvnet.classes.name';
              
            outputSize = layers{end}.NumClasses;
        otherwise
            warning('Unsupported layer type %s', l.type);
            
    end
    outputSize = layers{end}.forwardPropagateSize(outputSize);
end
    
layers = nnet.cnn.layer.Layer.createLayers(layers);
net = SeriesNetwork(layers);

%--------------------------------------------------------------------------
function layer = createInputImageLayer(n)

if numel(n.averageImage)==2    
    imageSize = [n.imageSize(1:2) 1];
else
    imageSize = n.imageSize(1:3);
end
normalizations = nnet.internal.cnn.layer.ZeroCenterImageTransform(imageSize);
emptyTransform = nnet.internal.cnn.layer.ImageTransform.empty;
if isfield(n,'averageImage')
    
    layer = nnet.internal.cnn.layer.ImageInput('input', imageSize, normalizations, emptyTransform);
    layer.AverageImage = single(n.averageImage);
else
    layer = nnet.internal.cnn.layer.ImageInput('input', imageSize, emptyTransform, emptyTransform);
end

%--------------------------------------------------------------------------
function layer = createLocalMapNorm2DLayer(l)
name = l.name;
windowMapSize = l.param(1);
alpha = l.param(3);
beta = l.param(4);
k = l.param(2);

layer = nnet.internal.cnn.layer.LocalMapNorm2D(name, windowMapSize, alpha, beta, k);

%--------------------------------------------------------------------------
function layer = createDropoutLayer(l)

layer = nnet.internal.cnn.layer.Dropout(l.name, l.rate);

%--------------------------------------------------------------------------
function layer = createPoolLayer(l)
name     = l.name;
poolSize = l.pool;
stride   = l.stride;
padding  = getPadding(l);

switch l.method
    case 'max'
        layer = createMaxPool(name, poolSize, stride, padding);
        
    case 'avg'
        layer = createAvgPool(name, poolSize, stride, padding);
        
    otherwise
        error('Unsupported pooling layer');
end

%--------------------------------------------------------------------------
function tf = isConv2DLayer(l, inputSize)
% inputSize: input size to the layer. If it equals size of filter then
% assume this is a fully connected layer.
sz = size(l.weights{1});
if isequal(sz(1:2),[1 1])
    tf = false;
else
    if inputSize(1:3) == sz(1:3)
        tf = false;
    else
        tf = true;
    end
end

%--------------------------------------------------------------------------
function layer = createReLULayer(l)
layer = nnet.internal.cnn.layer.ReLU(l.name);

%--------------------------------------------------------------------------
function layer = createFullyConnectedLayer(l)

[name, filterSize, numChannels, numFilters, ~, ~, W, B] = getConvLayerParams(l);

inputSize  = [filterSize numChannels];
outputSize = numFilters;

layer = nnet.internal.cnn.layer.FullyConnected(name, inputSize, outputSize);

outputSize = numFilters;

% Fully connected layers should take their weights in 4-D format. 
layer.Weights.Value = W;
layer.Bias.Value = reshape(B,1,1,outputSize);

%--------------------------------------------------------------------------
function [name, filterSize, numChannels, ...
    numFilters, stride, padding, W, B] = getConvLayerParams(l)

name = l.name;

if isscalar(l.stride)
    stride = [l.stride l.stride];
else
    stride = l.stride;
end

padding = getPadding(l);

if isfield(l, 'weights')    
    W = l.weights{1};
    B = l.weights{2};       
else    
    W = l.filters;
    B = l.biases;    
end

szW = size(W);

filterSize = szW(1:2);
numChannels    = szW(3);

if ndims(W) > 3
    numFilters = szW(4);   
else
    numFilters = 1;
end

%--------------------------------------------------------------------------
function layer = createConv2DLayer(l, inputSize)

[name, filterSize, numChannels, numFilters, stride, padding, W, B] = getConvLayerParams(l);

inputNumChannels = inputSize(end);

if numChannels < inputNumChannels    
    
    numGroups = inputNumChannels/numChannels;
    assert(numGroups == 2, 'Filter grouping must be 2');
    
    numFilters = repelem(numFilters/numGroups, 1, numGroups);      
end

layer = nnet.internal.cnn.layer.Convolution2D(name, filterSize, ...
    numChannels, numFilters, stride, padding);

% set weights and biases
layer.Weights.Value = W;
layer.Bias.Value = reshape(B,1,1,[]);

%--------------------------------------------------------------------------
function padding = getPadding(l)
% Padding in MatConvNet is defined as [TOP BOTTOM LEFT RIGHT]. 
padding = l.pad;
ph = padding(1:2);
pw = padding(3:4);

% only support symmetric padding
assert(ph(1) == ph(2), 'Only symmetric padding is supported');
assert(pw(1) == pw(2), 'Only symmetric padding is supported');

padding = [ph(1) pw(1)];

%--------------------------------------------------------------------------
function layer = createAvgPool(name, size, stride, padding)
layer = nnet.internal.cnn.layer.AveragePooling2D( ...
    name, ...
    size, ...
    stride, ...
    padding);

%--------------------------------------------------------------------------
function layer = createMaxPool(name, size, stride, padding)
layer = nnet.internal.cnn.layer.MaxPooling2D( ...
    name, ...
    size, ...
    stride, ...
    padding);
