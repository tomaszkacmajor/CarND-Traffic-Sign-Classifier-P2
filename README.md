## Traffic Sign Recognition Program
In this project, deep neural networks are used to classify traffic signs. The model is trained so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the training, the model program is tested on new images of traffic signs found on the web. 

The working code in Python along with partial outputs is presented in [Jupyter Notebook file](Traffic_Sign_Classifier.ipynb) or equivalent [HTML document](Traffic_Sign_Classifier.html).

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

### Data Set Summary & Exploration
To calculate summary statistics of the traffic signs data set the [Numpy](http://www.numpy.org/) library is used:

The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 
![alt text][Readme resources/ImagesTrainingSet.png]
This bar chart is showing how many images there are in the traninig set f 
![alt text][Readme resources/HistogramTrainingSamples.png]
Below, there are signs from the same label to have an idea of how different they can be (various light conditions, different angles etc.)
![alt text][Readme resources/ImagesSameType.png]

### Data preprocessing
I decided to generate additional data because in the chart above it can be seen that many traffic signs are underrepresented comparing to the most frequent ones. To obtain the data balance, I generated additional images so that each label has about 3200 samples (which is 4 times the mean number of samples per label before this operation). Such data augmentation provides also additional information to the model because while genereating images I randomly rotated and changed the brightness of images (using OpenCV library).
Below, there are some examples of new images generation. 
![alt text][Readme resources/TestImageToTransform.png]
![alt text][Readme resources/TestImagesAfterTransformation.png]
The new training data set has been increased from 34799 to 139148 samples. The validation and test set remained untouched. 

As a next step, I decided to convert the images to grayscale because it makes 3 times less the data which strongly influences on the training time. In addition, it has been shown in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) describing analogous problem that such operation helps in obtaining better results. 
I also normalized the image data to be between -1 and 1. It prevents from the numerical unstabilities which can ocuur when the data resides far away from zero. 

Here is an example of a traffic sign image before and after grayscaling and normalization. Below, the histograms of both images are depicted. 
![alt text][Readme resources/Grayscaling.png]
![alt text][Readme resources/HistogramAfterGreyAndNorm.png]

### Design and Test a Model Architecture
My final model architecture is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) convolutional neural network which has been proved to work on many similar problems. 
## Traffic Sign Recognition Program
In this project, deep neural networks are used to classify traffic signs. The model is trained so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the training, the model program is tested on new images of traffic signs found on the web. 

The working code in Python along with partial outputs is presented in [Jupyter Notebook file](Traffic_Sign_Classifier.ipynb) or equivalent [HTML document](Traffic_Sign_Classifier.html).

The goals / steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

### Data Set Summary & Exploration
To calculate summary statistics of the traffic signs data set the [Numpy](http://www.numpy.org/) library is used:

The size of training set is 34799
The size of the validation set is 4410
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 
![alt text][Readme resources/ImagesTrainingSet.png]
This bar chart is showing how many images there are in the traninig set f 
![alt text][Readme resources/HistogramTrainingSamples.png]
Below, there are signs from the same label to have an idea of how different they can be (various light conditions, different angles etc.)
![alt text][Readme resources/ImagesSameType.png]

### Data preprocessing
I decided to generate additional data because in the chart above it can be seen that many traffic signs are underrepresented comparing to the most frequent ones. To obtain the data balance, I generated additional images so that each label has about 3200 samples (which is 4 times the mean number of samples per label before this operation). Such data augmentation provides also additional information to the model because while genereating images I randomly rotated and changed the brightness of images (using OpenCV library).
Below, there are some examples of new images generation. 
![alt text][Readme resources/TestImageToTransform.png]
![alt text][Readme resources/TestImagesAfterTransformation.png]
The new training data set has been increased from 34799 to 139148 samples. The validation and test set remained untouched. 

As a next step, I decided to convert the images to grayscale because it makes 3 times less the data which strongly influences on the training time. In addition, it has been shown in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) describing analogous problem that such operation helps in obtaining better results. 
I also normalized the image data to be between -1 and 1. It prevents from the numerical unstabilities which can ocuur when the data resides far away from zero. 

Here is an example of a traffic sign image before and after grayscaling and normalization. Below, the histograms of both images are depicted. 
![alt text][Readme resources/Grayscaling.png]
![alt text][Readme resources/HistogramAfterGreyAndNorm.png]

### Design and Test a Model Architecture
My final model architecture is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) convolutional neural network which has been proved to work on many similar problems. 

The model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16  				|
| Fully connected		| inputs 400, outputs 120        									|
| RELU					|		
| Fully connected		| inputs 120, outputs 84        									|
| RELU					|		
| Fully connected		| inputs 84, outputs 10        									|
| Softmax				|        									|


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer	Description
Input	32x32x3 RGB image
Convolution 3x3	1x1 stride, same padding, outputs 32x32x64
RELU	
Max pooling	2x2 stride, outputs 16x16x64
Convolution 3x3	etc.
Fully connected	etc.
Softmax	etc.
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

training set accuracy of ?
validation set accuracy of ?
test set accuracy of ?
If an iterative approach was chosen:

What was the first architecture that was tried and why was it chosen?
What were some problems with the initial architecture?
How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Which parameters were tuned? How were they adjusted and why?
What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
If a well known architecture was chosen:

What architecture was chosen?
Why did you believe it would be relevant to the traffic sign application?
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

alt text alt text alt text alt text alt text

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Image	Prediction
Stop Sign	Stop sign
U-turn	U-turn
Yield	Yield
100 km/h	Bumpy Road
Slippery Road	Slippery Road
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Probability	Prediction
.60	Stop sign
.20	U-turn
.05	Yield
.04	Bumpy Road
.01	Slippery Road
For the second image ...

(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16  				|
| Fully connected		| inputs 400, outputs 120        									|
| RELU					|		
| Fully connected		| inputs 120, outputs 84        									|
| RELU					|		
| Fully connected		| inputs 84, outputs 10        									|
| Softmax				|        									|


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer	Description
Input	32x32x3 RGB image
Convolution 3x3	1x1 stride, same padding, outputs 32x32x64
RELU	
Max pooling	2x2 stride, outputs 16x16x64
Convolution 3x3	etc.
Fully connected	etc.
Softmax	etc.
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

training set accuracy of ?
validation set accuracy of ?
test set accuracy of ?
If an iterative approach was chosen:

What was the first architecture that was tried and why was it chosen?
What were some problems with the initial architecture?
How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Which parameters were tuned? How were they adjusted and why?
What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
If a well known architecture was chosen:

What architecture was chosen?
Why did you believe it would be relevant to the traffic sign application?
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

alt text alt text alt text alt text alt text

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Image	Prediction
Stop Sign	Stop sign
U-turn	U-turn
Yield	Yield
100 km/h	Bumpy Road
Slippery Road	Slippery Road
The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Probability	Prediction
.60	Stop sign
.20	U-turn
.05	Yield
.04	Bumpy Road
.01	Slippery Road
For the second image ...

(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?