#Self Driving Car: Detecting Lane Lines

## Canny Edge Detection and Hough Transform with and OpenCV

### *Daniel Wolf*

### **Introduction**

The goal of this project is to detect lane lines on the highway using image
analysis techniques.
My video result is on YouTube [here](https://www.youtube.com/watch?v=n1lDA-ALE3c).

The steps I took are the following:

* Convert image to grayscale and use Gaussian blur
* Apply Canny edge detection
* Implement region masking for targeted lane detection
* Use Hough transform to identify the lane lines among the detected edges
* Extrapolate lane lines and draw on the final image

[//]: # (Image References)

[image1]: ./examples/pipeline.jpg
[image2]: ./examples/result.jpg

### **Pipeline Overview**

The code for defining the pipeline functions and creating the pipeline is in 
code cells 3 and 4 of the Jupyter notebook.  

I start by converting the image to grayscale and using Gaussian blur to 
smooth the pixels. The OpenCV `Canny()` Edge Detector is then applied to detect all of the edges.
Canny edge detection intends to accurately capture as many edges in the image
as possible, so there are many more edges captured than just the lane line
edges. For this reason, I then apply the `Hough()` transform function. The Hough transform
identifies which edges are most likely to be lane lines by determining how many lines 
intersect in Hough space. I used a line intersection threshold of 75 
to reduce the number of false positives. I also implemented region masking
in order to focus in on the areas of the image that are likely to hold lane 
lines. Finally, I used the Hough lines to extrapolate to lane lines that I could
draw on the final image through an algorithm that averages slopes and x- and y-coordinates.

The image below shows the pipeline result at each stage described above:

![][image1]


Here is a larger image of the resulting output:

![][image2]



### **Video Result**

When processing a video with consecutive images, I improved the robustness of my pipeline 
by discarding any line that does not meet the required number of Hough space 
intersections. In these scenarios, the detected lines from the prior frame are used.

Here's a [link to my video result](https://www.youtube.com/watch?v=n1lDA-ALE3c).


### **Conclusion**

While my current implementation displays lane lines with a reasonably high level 
of accuracy, there is some room for improvement. For example, I could use 
my Lines class to smooth out the lane lines from frame to frame so that they 
are less jittery. This implementation does not work as well with curved lanes
or with shadows. I take more advanced approaches to finding lane lines as part of
the Advanced Lane Finding in project 4.

