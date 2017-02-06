##Advanced lane finding report

**The goals / steps of this project are the following:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cheeseboard.png "Undistorted"
[image2]: ./output_images/distortion_corrected.png "Road Transformed"
[image3]: ./output_images/binary_image.png "Binary Example"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/slid_win_fit.png "Fit Visual"
[image6]: ./output_images/lane_on_image.png "Output"
[image7]: ./output_images/binary_image_per_trans.png "Output"

---

###Here I will describe how I addressed each [Rubric point](https://review.udacity.com/#!/rubrics/571/view) in my implementation.  

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  I use OpenCV 'cv2.findChessboardCorners' function to find the "imgpoints". 

once the  `objpoints` and `imgpoints` are ready, I use OpenCV 'cv2.calibrateCamera' to compute the camera calibration and distortion coefficients.  We can now apply the camera matrix and distortion coefficients to correct distortion effects on camera input images using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 143 through 151 in `lane.py` file).  
For color, I convert the RGB image to the HLS color space, and then use the S channel becasue it is very efficient for picking out lane lines under different color and contrast conditions. The binary image was obtained by setting the S chanel value between 170 and 255 to one and the left to zero.
For gradient, I convert the  input RGB image to grayscale. I then apply a Sobel filter in the X direction to get image edges that match the direction of the lane lines. The grey scale image then be converted to binary images by setting the scaled abslute gradient between 20 and 100 to one, and the lest to zero.

Here's an example of my output combining the above two thresholds.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform at lines 154 through 160 in `lane.py` file with a function named `pers_trans`.  
The `pers_trans` function takes as inputs an image `img`, as well as source `src` and destination `dst` points.  

I apply a perspective transform on the image to generate an image with the effect of looking down on the road from above. I have defined a preset coordinate list, i.e., the source `src` and destination `dst` points, to use for the perspective transformation,
and then use OpenCV `cv2.getPerspectiveTransform()` function generates a perspective transform matrix.

```
        # Source points coords for perspective xform
        self.src = np.float32(
            [[(self.width / 2) - 55, self.height / 2 + 100],
            [((self.width / 6) + 40), self.height],
            [(self.width * 5 / 6) + 100, self.height],
            [(self.width / 2 + 70), self.height / 2 + 100]])
        # Dest. points coords for perspective xform
        self.dst = np.float32(
            [[(self.width / 4), 0],
            [(self.width / 4), self.height],
            [(self.width * 3 / 4), self.height],
            [(self.width * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 253, 720      | 320, 720      |
| 1166, 720     | 960, 720      |
| 710, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image, and its warped counterpart after perspective transform using the OpenCV `cv2.warpPerspective()` function to verify that the lines appear parallel in the warped image. I use the OpenCV `cv2.warpPerspective()` function to do this.

![alt text][image4]

Here I also give a example of the perspective tranform applie on the binary threshold image above.

![alt text][image7]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code used to identify and fit the lane is the function `init_lane_locate` on line 203, the function `lane_locate` on line 262 and function `lane_fit` on line 280 in `lane.py` file.

To identify the first frame, or for cases where we lanes are ‘lost’ (i.e. we have not been able to reliably detect a ‘good’ lane for a number of frames), I generate a histogram of the bottom half of the image. Then using the two peaks in this histogram, I determine a good starting point to start searching for image pixels at the bottom of the image. 

Once these points are calculated, I divide the image into 9 horizontal strips of equal size. For the bottom strip, I mask out everything outside of a small window in order to extract the pixels that belong to the lane, effectively discarded all other ‘noise’ pixels.

I repeat this process for each strip, using histograms on the strips to determine good lane pixes, and then the coordinates of the left and right lane. Once I have processed all strips, I then am left with images for both the left and right lanes. 

For all other frames, I use the polynomial lane equation for the previous frame (caluculated in the stage below) to estimate the x coordinates of the left and right lane. This avoids the expensive computation and speeds the process a lots, because we take advantage of the previous information of the lane location.

Once we get the coordinates of the lane pixes, we can fit a line with `np.polyfit` function. Then I do two checks to determine if the calculated lane if ‘good’ or not. 

- First I check to see that the Radius of Curvature of the lane is above a minimum threshold of 587 meters, which is from the U.S. government specifications for highway curvature.

- The second check I do is to see that the x coordinates of the left and right lanes. It should not change too much from one frame to anoter condiersing that there is 25 frames per second.I found that checking for a 15 pixel delta worked well for this check.

- If any of the above two checks fail for a lane or two, I consider the lane ‘lost’ or undetected for that frame, and I use the average values from the previous n frame.  If a lane has not been successfully detected for 5 successive frames, then I trigger a full scan for detecting the lanes in the Locate Lanes section. I always mantian a best fit by averaing 5 'good' fit (including current one, if detected 'good') and use it for current lane drawing.

Here’s what the fitted lanes image looks like:

![alt text][image5]


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code used to calculate the radius of the curvature and offsets of the vechile with respecitive to the lane center is the function `get_curvature_offsets` on line 170 throgh 200 in `lane.py` file.

To calculate the curvature of the lane, we have to fit the line in the real world space by conveting the pixes coordinates to distance in meters. After fitting a line, it is pretty easy to calculate the radius of the curvature.

To calculate the offsets of the vehicle with repective to the lane center, we suppose the camera is located in the middle bottom of the image. We can also caiculate the x coordinates of the lane centers by take the average of the bottom x coordinates of the left and right lanes, which is easily to obtain by putting the maximum y coorinates to the fitted polynomial line of both lanes. By comparing the average x coordinates of the lane center with the camera location, we can get the offsets in pixes space, and then convet it to real world space by meters per pixel in x dimension.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Aftet applied the above techniques on a single image,the lanes have been detected in the perspective view, and we need a inverse perspective transform to draw the detected lanes onto the undistorted image.

I also annotate the average Radius of Curvature of lanes and the vehicle offsets with respective to the lane center. Finally, I calculate a value for how far the car is from the center of the lane and annotate this too. The final output looks like this:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a Youtube link to my fianl prejection video solution.
[![IMAGE ALT TEXT](http://img.youtube.com/vi/_k94oNCMjl0/0.jpg)](https://youtu.be/_k94oNCMjl0 "a small networkd ")

---

###Reflection

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The above pipline works well on the project video, but not on the challenge video because of the severe lighting condition change, and road color and texture change. I would need to do more work on the Binary Thresholding stage to help with light variances and high contrast patterns in the road surfaces; and condiser and implement more checks for the fitted lanes, etc.  

