Cats Vs Dogs Classification With Matlab

Performing Transfer Learning on 200 Images: 100 dog images, 100 cat images

Partitioning Data:
60% Training, 20% Cross Validation, 20% Testing
Note - Data will be partiotioned 80/20 to begin, and 80% will be used by the
Classification Learner App. The classification learner app will then partition
the 80% of data into two sets -> 75% Training and 25% Holdout Validation. This
will produce the desired partitions of 60/20/20
 
Using a PreTrained CNN to extract features from image data: AlexNet

Using Discriminant Analysis to train a model with the 4096 features provided by AlexNet.

Part I: Selecting Image Pre-Processessing Strategy

Model 1) 200 images with no pre-processing filters
A) Linear Discriminant Cross Validation resulted in 92.5% Accuracy
Linear Discrimant Classification on Test Set resulted in:
confMat =
    1.0000         0
    0.2500    0.7500

B) Quadratic Discriminant Cross Validation resulted in 92.5% Accuracy.
confMat =
    1.0000         0
    0.2500    0.7500


Model 2) 200 images after applying gaussian blur to each image
Gaussian Blur standard deviation value = 0.5 (default)
Linear Discriminant Cross Validation resulted in 90%
Linear Discriminant Classification on Test Set resulted in:
confMat = 
    1.0000         0
    0.0500    0.9500

Quadratic Discriminant Cross Validation resulted in 90%
Quadratic Discriminant Classification of Test Set resulted in:
confMat = 
    1.0000         0
    0.1500    0.8500



Based on the results from Part I
We will increase data set to 1000 images, 500 cats and 500 dogs
We will apply gaussian blur to images before extracting features,
because Test Results were best when using the gaussian blur filter


Part II: Increasing data set to 1000 images, 500 cat and 500 dog

Model 1) 1000 images after applying gaussian blur to each image
Gaussian Blur standard deviation value = 0.5 (default)
Linear Discriminant Cross Validation resulted in 94.5%
Linear Discriminant Classification on Test Set resulted in:
confMat =
    0.9800    0.0200
    0.1000    0.9000

Quadratic Discriminant Cross Validation resulted in 92.5%
Quadratic Discriminant Classification of Test Set resulted in:
confMat = 
    0.9700    0.0300
    0.1300    0.8700
