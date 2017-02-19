#Self Driving Car: Vehicle Detection

## Computer Vision with OpenCV

### *Daniel Wolf*

## **Introduction**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction 
on a labeled training set of images and train a classifier Linear SVM classifier
* Append additional features using color transform and binned 
color features, as well as histograms of color 
* Implement a sliding-window technique and use a trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected

[//]: # (Image References)
[image1]: ./examples/car_not_car.jpg
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.jpg
[video1]: ./project_video.mp4


### **Histogram of Oriented Gradients (HOG)**

The function that extracts HOG features is in code cell 2 of the Jupyter notebook. This function 
is called by `extract_features()` in code cell 5, which also extracts spatial features 
and color histogram features.

I read in all the `vehicle` and `non-vehicle` images in code cell 7.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:


![][image2]

### **Parameter Selection**

I took a robust trial and error approach to deciding on all parameters, 
including spatial binning and color histogram parameters in addition to HOG. 
The code for this experimentation is in code cell 9. I tried a range of values for
each parameter while keeping all other parameters constant to estimate an 
optimal value for each parameter. I also tried multiple combinations of HOG
parameters. I determined optimal performance by comparing the test accuracy
given by the Linear SVM 
classifier. I evaluated 206 combinations in total, and 44 of these had
a testing accuracy above 99%. Out of these 44, I chose the set of parameters that 
generated the shortest feature vector length in order to help with the speed of image 
processing for my video. The feature vector length for the combination of parameters shown here 
is 1,296.

**Final Parameter Selection**:

* color_space = 'LUV'
* HOG orientations = 9
* HOG pix_per_cell = 16
* HOG cell_per_block = 1
* hog_channel = 'ALL'
* spatial_size = (16, 16)
* hist_bins = 32
* spatial_feat = True
* hist_feat = True
* hog_feat = True



### **Feature Extraction and SVM Classification**

As mentioned above, I trained a linear support vector machine classifier, as shown in code cell 14.
This was implemented using sklearn's [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
function.

Feature extraction is called in code cell 11, and the features are stacked 
and scaled to zero mean and unit variance (with sklearn `StandardScaler`) in code cell 13. 
Scaling is critical for this project since the features include spatial binning and color histogram 
features along with the HOG features, and each has different ranges of values. 
Also in this cell, I defined the labels vector and 
split the data into training and testing sets before training.


### **Sliding Window Search**

My sliding and searching window functions are in code cells 16, 17, 18. The 
first function `slide_window()` defines the ranges for each window that 
will be searched. The second function `search windows` extracts features 
from each window and classifies them as car or not car. The third function
`window_pipeline()` defines the window sizes, ranges, and overlaps and then 
calls the first two functions.

Since the number of windows has a significant impact on processing time,
I chose my windows carefully. I applied a smaller window size to 
the middle of the image since that is where cars will be farther away.
I applied the largest window size to nearly half of the image
so that it could capture larger cars in the foreground. Here are the window 
sizes used along with the ranges and overlaps:


| Window Size | Y start/stop | X start/stop  | XY overlap |
|:-----------:|:------------:|:-------------:|:----------:|
| 64x64       | (415 , 500)  | (150 , 1050)  | (0.8, 0.8) |
| 88x88       | (395 , 515)  | (150 , 1050)  | (0.8, 0.8) |
| 126x126     | (400 , 540)  | (None , None) | (0.8, 0.8) |
| 276x276     | (400 , 690)  | (None , None) | (0.8, 0.8) |

This combination of window sizes and ranges results in 497 windows 
that are searched. I felt very comfortable with this implementation
because it could accurately identify cars while still keeping 
the video processing time around 1.2 seconds per iteration.
Here is an example image that shows a box for all overlapping windows 
that are searched:

![][image3]

### **Optimization and Reducing False Positives**

The features that I selected consistently result in >99% testing accuracy,
so I felt confident that my Linear SVM classifier was reliable for 
identifying cars in images. As described above, I went through a trial and error
process with a wide range of features. I did test adjustments to the 
threshold for the decision function, but I found that 0 achieves sufficient overall performance.

Beyond the classifier itself, optimizing performance consisted of reducing the processing time and
removing false positives through window thresholding. I reduced the processing time by minimizing the 
number of features and the number of searched windows, as mentioned above.
Secondly, I implemented two thresholds to make it very difficult
for false positives to be drawn onto the final output. The first of these
utilizes a heatmap approach such that any particular area of an image frame 
must have at least 3 overlapping "positive car" windows for it to be identified
as a car for that image frame. It is in code cell 20 for single images and code cell
24 for video processing. This reduces the likelihood of false positive
considerably, because the false positive would have to occur 3 times in the same
area of the image. I explain the second threshold for removing false positives in #2 
of the Video Processing section.

Here is an example of the first threshold. Note that 
as part of the threshold process, the bounding boxes are simplified to 
a single bounding box for each car. This is explained further below.

![][image4]



---

### **Video Implementation**

Here's a [link to my video result on YouTube](https://youtu.be/zTn4VE_5Ns8).

### **Improving Robustness**

I recorded the positions of positive detections in each frame of the video. 
From the positive detections I created a heatmap and then thresholded 
that map to identify vehicle positions.  This represents the first
threshold mentioned above. I then used `scipy.ndimage.measurements.label()` to 
identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected. All of these 
steps are done in code cell 24 of the Jupyter notebook.

The second threshold to remove false positives relates directly
to video processing. It requires that a heatmap 
appear on at least 9 out of 10 consecutive image frames for it to be
drawn onto the final output. It is also in code cell 24. With my classifier, it is very rare for a false
positive to occur on this many consecutive images.

I also spent some time creating an algorithm that tracked 
bounding boxes across multiple frames using two class variables and a function
called `average_bbox()` in code cell 23. This helped with determining when a
bounding box should be added or removed based on whether it the new bounding box
is close to any prior bounding boxes that are currently drawn onto the output. 
This also helped with smoothing the boxes from frame to frame.

### **Video frame examples**

Here are six consecutive frames with all "positive car" windows, their corresponding heatmaps, and their
corresponding bouding boxes that are derived by `scipy.ndimage.measurements.label()`. As mentioned
above, a threshold is implemented such that a bounding box only appears on the final output 
if 1) there are at least 3 overlapping car windows and 2) the heatmap covers the same area
on at least 9 out of 10 consecutive frames.

![][image5]

---

### **Conclusion**

My strategy in this project was to minimize the 
processing time while still accurately tracking the cars, which I did by 
using a relatively small number of features and finding a precise window search area.

With this approach, my algorithm may not work 
as well on sharp turns or on hills, because cars may fall outside of the searchable
range that I implemented. This demonstrates a tradeoff between processing time and the amount of the
image that is searched.

Also, my approach was to treat false positives very strictly, so that by the time
a car is identified on the final output, it has passed two thresholds as mentioned above.
Fltering out false positives with this robust approach
will filter out true positives until the thresholds are reached, so even though
it works well on the highway with cars steadily passing and getting passed, it may 
not work as well with a car entering the camera image very quickly (for example
in a dangerous situation).

In addition to making the project more robust by resolving the issues mentioned 
above, I could enhance this project by adding additional features, such as
the approximate distance between my car and each detected car. I could also include
lane tracking features that I implemented in the last project. So much potential!

#### Key Installations

* Python 3
* Open CV


