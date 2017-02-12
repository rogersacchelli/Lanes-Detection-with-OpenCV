#Advanced Lane Finding

##Goals 

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


###Camera Calibration

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


### Perspective Transform
Perspective transform helps to adjust the difference between real and perceived distance from objects far from camera. Without perspective transform would not be possible to precisely draw the lane line correctly.


**Source and Destination Points**
To achieve a precise transformation, we need to specify source and destination points, through a clock-wise sequence starting from first quadrant.
![](https://fivedots.coe.psu.ac.th/~ad/jg/ch139/persp2.jpg)

**Offset**
The offset indicated below is a trick to bring left and right lines closer to each other in order to not loose line curvature information.

This trick increases the number of points which later will be calculated to fit to second order polynomial function, to represent the line.

|Point | Source Points (x,y) | Destination Points (x,y) |
|-| ------------- | ------------- |
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
              

### Detecting Lane Pixels

