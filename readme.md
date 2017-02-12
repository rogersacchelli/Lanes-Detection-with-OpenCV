# Advanced Lane Finding

## Goals 

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation 
of lane curvature and vehicle position.

| Raw Input  | Final Output |
| ------------- | ------------- |
| ![Raw](http://i.imgur.com/bc60myS.png)   | ![Undistorted](http://i.imgur.com/W6sxa33.png)  |


### Camera Calibration

This is a fundamental step of the project, since without calibration, the image analisys may fall into uncorrect results.

Camera calibration is performed by opencv in two steps:

1. Find Chess Board Corners
	* This function retutns radial and tangential transform parameters
	* [OpenCV Chess Board Corners](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners) 
2. Camera Calibration
	* Calibrate Camera function takes as arguments the output of Chess Board Corners Function, plus object points values to return the Calibration Function.
	* [OpenCV Calibrate Camera](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera)

**Undistorted Image Example:**

| Before  | After |
| ------------- | ------------- |
| ![Distorted](http://i.imgur.com/HAcV7QF.jpg)   | ![Undistorted](http://i.imgur.com/HBHPVhC.jpg)  |

***Find in the code: main.py:44***

 <hr>

### Thresholded Binary  Image

Thresholding is applied after image ungoes to [Sobel](http://docs.opencv.org/3.1.0/d5/d0f/tutorial_py_gradients.html) gradient transform on x axis, which means we're interested mainly on vertical lines. Although before applying sobel transform, image it converted to gray scale and to HSV color space. The reason for such transformation is that, changing color spaces, makes easier to detect lines of different color, which causes a better result for Sobel transform.

	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = hls[:, :, 1]
    s_channel = hls[:, :, 2]


    # Binary matrixes creation
    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    s_binary = sobel_binary
    combined_binary = s_binary.astype(np.float32)

    # Sobel Transform
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = 0 #cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.abs(sobelx**2 + sobely**2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))

    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1

    # Threshold color channel
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds

    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
    
**Binary Output**
![](http://i.imgur.com/NZuNecy.jpg)

#### Masking
To reduce the amount of undesired points during image processing, after image thresholding the image is masked to output only the region of interest. This technique improves success of detection.

***Find in the code:*** 
1. Binary Thresholding:	main.py:62 - image_binary():177-202
2. Image masking: main.py: 62  - image_binary():208:224


### Perspective Transform
Perspective transform helps to adjust the difference between real and perceived distance from objects far from camera. Without perspective transform would not be possible to precisely draw the lane line correctly.


**Source and Destination Points**
To achieve a precise transformation, we need to specify source and destination points, through a clock-wise sequence starting from first quadrant.
![](https://fivedots.coe.psu.ac.th/~ad/jg/ch139/persp2.jpg)

**Offset**
The offset indicated below is a trick to bring left and right lines closer to each other in order to not loose line curvature information.

This trick increases the number of points which later will be calculated to fit to second order polynomial function, to represent the line.

|Point | Source Points (x,y) | Destination Points (x,y) |
|------| ------------- | ------------- |
|A| (595, 452)  | ((Source Xd + offset), 0) |
|B| (685, 452)  |  ((Source Xc - offset), 0)  |
|C| (1110, y_size)  | ((Source Xc - offset), y_size)|
|D| (220, y_size)  | ((Source Xd + offset), y_size) |

		line_dst_offset = 200
		src = [595, 452], \
		      [685, 452], \
		      [1110, img_b.shape[0]], \
		      [220, img_b.shape[0]]
        
        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]
              

***Find in the code:*** 
1. Perspective transform: main.py:65:81

### Detecting Lane Pixels

The binary image obtained allied to perspective transformed image allow us to determine where lane lines are placed.

To start the search for lanes position, an histogram from the binary image is generated. The histogram quantified the number of pixels along 'x' axis, as a consequence peaks found along the histrogram points to lane lines. See below the whole process from obtaining the undistorded image to its histogram.

![](http://i.imgur.com/I8f3dfs.jpg)


* Upper left:Undistorted Image
* Upper right: Binary masked image
* Lower left: Perspective transformed from binary masked image
* Lower right: Histogram from binary transformed image

#### Fitting lines to a polynomial function
As it's possible now to detect the major concentrarion of lines, a technique called sliding window is used to map the group of pixels that forms the lane line and then fit a second order polynomial.
![](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588cf5e0_screen-shot-2017-01-28-at-11.49.20-am/screen-shot-2017-01-28-at-11.49.20-am.png)

	# Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_w.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_w.shape[0] - (window + 1) * window_height
        win_y_high = img_w.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

***Find in the code:*** 
1. Fit lines - fit from already found fit - main.py:102
2. Fit lines - fit from unknown (sliding_windown) - main.py:108
3. Averaging fit - main.py:110:118
 

### Lane Curvature and Off Center distance


Lane curvature is calculated throught the following relation:

$ Radius = \frac{[1+(\frac{dx}{dy})^2]^{3/2}}{|\frac{d^2x|}{dy^2}} $

Another important information for the calculation is the representation of each pixel in meters on the image:

> Pixel x axis = 3.7 / 700 m
> Pixel y axis = 30 / 720 m


	ploty = np.linspace(0, img_height - 1, img_height)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius = round((float(left_curverad) + float(right_curverad))/2.,2)
    

#### Off Center distance

Since we know the x axis pixel/distance relation and we also know the distance in pixels beween left and right lane, we can calculate the distance the car is from center to the left or to the right, considering that our cam is positioned right in the middle of image.

**Center between lane lines**

$ line_{center} = (right_{fit}[2] - left_{fit}[2]) / 2 $, where:

* $right_{fit}[2]$ and $left_{fit}[2]$, are the linear coeficients from fit functions. They represent where the polynomia function crosses x axis where y = 0;

** Distance in pixels between car center and lanes center**

$offcenter_{pixels} = (line_{center} - car_{center})$

In meters:

$offcenter_{meters} = offcenter_{pixels} * (3.7 / 700)$


***Find in the code:*** 
1. main.py:129
2. draw_lines():382:421

### Inverse Perspective Tranformation

A important step even before lane radius calculation is permforming inverse perspective to transform to plot the image on its real shape.

The function is the same from the first perspective transform, but now source and destionation points are inverted on its arguments call.


***Find in the code:*** 
1. main.py:129
2. draw_lines():371, 378

### Video Demostrantion

[![ ](http://img.youtube.com/vi/iFhYH4QPJ9A/0.jpg)](http://www.youtube.com/watch?v=iFhYH4QPJ9A "Advanced Lane Detection")

### Discussion

After completing the basic requirement, two main subjects stand out.

1. Performance:
The final performance is not as fast as I wanted to be, the fps of roughly 12 fps is not enough for real time scenarios. During processing, two important steps takes almost 50 ms to complete, which gives the processing an starting point of 20 fps, which is already low fps. New improvements then must be made to improve unditortion function and binary transform. An interesting aproach would be porting this delopment from C++ to Python which suggests to speed up the undistortion 5x times:

	[How to speed up image undistortion for a video/image sequence?](https://shiyuzhao1.wordpress.com/2013/11/21/how-to-speed-up-image-undistortion-for-a-videoimage-sequence/)

2. Robustness
There's a lot to be done to make this framework robust for different envorinments. New videos are available to help this development and from this point beyond it should take much lower effort than bringing the development from zero to this point.

This project was very interesting which gave important skills on computer vision techniques.


