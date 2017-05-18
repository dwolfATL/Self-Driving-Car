#Self Driving Car: Advanced Lane Finding

## Computer Vision with OpenCV

### *Daniel Wolf*

## **Introduction**

The goal of this project is to detect lane lines on the highway using OpenCV.
My video result is on YouTube [here](https://www.youtube.com/watch?v=kC3Ez4lvkbI).
The steps I took are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/chessboard_undistorted.jpg "Calibration Undistorted"
[image2]: ./examples/undistorted_test2.jpg "Undistorted"
[image3]: ./examples/binary_test2.jpg "Binary Example"
[image4]: ./examples/warped_test2.jpg "Perspective Transform Example"
[image5]: ./examples/fitlines_test2.jpg "Fit Visual"
[image6]: ./examples/output_test2.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"


## **Camera Calibration**

The code for this step is contained in code cell 5 of the Jupyter notebook.  

I start by preparing "object points", which will be the (x, y, z) coordinates 
of the chessboard corners in the world. Here I am assuming the chessboard is 
fixed on the (x, y) plane at z=0, such that the object points are the same for 
each calibration image.  Thus, `objp` is just a replicated array of coordinates, 
and `objpoints` will be appended with a copy of it every time I successfully 
detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the 
corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera 
calibration and distortion coefficients using the `cv2.calibrateCamera()` 
function (code cell 6).  I applied this distortion correction to the test 
image using the `cv2.undistort()` function and obtained this result: 

![Calibration Undistorted][image1]

---

## **Pipeline (single images)**

I apply distortion correction to my test images as the first step in my gradient threshold pipeline,
which is in code cell 9. The same calibration is applied to these images, which results in an 
adjustment to the outer edges of the test images. An example undistorted image is shown here:

![Undistorted][image2]

I then used a combination of color and gradient thresholds to generate a binary image. The threshold pipeline is 
in code cell 9 and references helper functions in code cell 8. Here are the steps involved:

1) Gaussian blur smooths the image pixels
2) Apply Sobel directional gradients in the x and y directions using a grayscale image
3) Apply gradient magnitude and direction thresholds using a grayscale image
4) Apply s channel threshold using an HLS color space version of the image
5) Apply region masking to remove pixels outside of where we expect the lanes to be

All of the threshold components above generate a binary image that looks like this, as an example:

![Binary Example][image3]

The next step is perspective transform, which is in code cell 10 of my Jupyter notebook.  
The `perspective_transform()` function takes as inputs an image (`img`), as well 
as source (`src`) and destination (`dst`) points.  I hardcoded the following source 
and destination points:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 470      | 350, 0        | 
| 735, 470      | 930, 0        |
| 1140, 720     | 930, 720      |
| 210, 720      | 350, 720      |

I know that my perspective transform is working as expected because the lane lines are approximately parallel in the warped image, as shown here:

![Perspective Transform Example][image4]

My functions to identify lane pixels are in code cell 11. I split the test image into thirds vertically
and in half horizontally. By summing up the pixel values for all of the coordinates and
identifying which x-coordinate has the largest sum (i.e. "histogram" approach) in each section,
the function can identify the most likely x-coordinate for the lane in that section. Once the x-coordinate 
is identified, a "window" can be applied such that the function can identify all non-zero pixel 
values in that window and append to left lane and right lane points array. This is contained within the 
`find_lane_pts()` function.

Also in code cell 11, the `fit_lanes()` function will apply a polynomial fit to the lane point arrays.
This fit is then used to create a lane line that goes from the bottom to the top of the warped image.

An example result of lane finding and fitting is shown here:

![Fit Visual][image5]

I calculated the radius of curvature of the lane and the position of the vehicle with respect to center
in code cell 12 in my Jupyter notebook. I used the fitted lines from the prior step
to calculate the radius of curvature. Here is a 
[tutorial and formula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) 
for radius of curvature. 
I used an approximation of 30 meters per 720 pixels horizontally and 3.7 meters per 700 pixels vertically
in the calculations.

Regarding position, I identified the x-coordinates of the left and right lanes at the bottom of the image 
(closest to the car). I found the center of the lane using half the distance
between these two coordinates. I also know the car's position to be the center of the image since the 
camera is mounted on the center of the car's hood. Using the approximation of 30 meters per 720 pixels,
I could then find the approximate distance in meters from the center of the lane to the car's position.

Both of these metrics are displayed on the output images.

I implemented the next step of plotting the result back down onto the road in code cell 13.  The `draw_polygon()` function fills a polygon between the fitted lines
on the warped image. The `draw_output()` function performs a perspective transform on the polygon back to 
the original perspective, and overlays it on the original image (with some transparency).

Code cell 14 uses a pipeline approach to iterate through all of the steps mentioned above, and here is an example of my result on a test image:

![Output][image6]

---

## **Pipeline (video)**

When processing a video with consecutive images, I improved the robustness of my pipeline by checking
that the lines are 1) approximately parallel and 2) far enough apart. If either of these checks fail,
I implemented a `Line()` class so that the frame can simply use the same lane lines as the prior frame. 

Also, I smoothed the edges of the polygon by averaging the x-coordinates of the last 10 frames, which 
helped them from becoming too wobbly with each frame. The line checking and averaging function 
is in code cell 18.

Here's a [link to my video result](https://youtu.be/kC3Ez4lvkbI).

---

## **Conclusion**

The two most challenging aspects were 1) defining source points for the perspective transform and 2) applying
the right thresholds to correctly identify lane pixels as reliably as possible. Each task was quite manual
and involved trial and error, so I would like a more robust approach for these aspects.

There are some occasions where my algorithm may not properly identify the lane lines. My lane lines may fail
in scenarios scenarios with abnormal lighting. I 
would need to experiment with different gradient thresholds and potentially different color spaces to 
maximize the pipeline's ability to identify lane pixels. I would also try to implement more effective 
region masking as I found that "noise" within the lane sometimes affected the lane lines, and I only 
masked areas outside of the lanes.

A second challenge for my algorithm is windy roads, because if the lane curves too sharply, 
it may fall outside of the "sliding window" after the perspective transformation. A stronger
implementation might utilize the prior frame's location of the lane to more accurately predict
the lane for the next frame.

#### Key Installations

* Python 3
* OpenCV

*Udacity Self Driving Car Nanodegree Term 1 Project 4, January 2017*

