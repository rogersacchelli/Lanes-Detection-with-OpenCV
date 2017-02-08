#Advanced Lane Finding

##Goals 

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

###Camera Calibration

This is a fundamental step of the project, since without calibration, the image analisys may fall into uncorrect results.

Camera calibration is performed in two steps:

1. Find Chess Board Corners
	* This function retutns radial and tangential transform parameters
	* [OpenCV Chess Board Corners](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#drawchessboardcorners) 
2. Camera Calibration
	* Calibrate Camera function takes as arguments the output of Chess Board Corners Function, plus object points values to return the Calibration Function.
	* [OpenCV Calibrate Camera](http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera)

**Undistorted Image Example:**

| Before  | After |
| ------------- | ------------- |
| ![Distorted](https://i.imgsafe.org/5feb6d664f.jpg)   | ![Undistorted](https://i.imgsafe.org/5fedaac69d.jpg)  |





