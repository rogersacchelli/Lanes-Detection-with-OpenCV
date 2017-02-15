import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line:
    def __init__(self):
        self.right_detected = False
        self.left_detected = False
        self.left_inside_detected = False
        self.left_outside_detected = False
        self.right_inside_detected = False
        self.right_out_detected = False

        self.radius = 0.0
        self.turn_side = 0
        self.left_fit = 0
        self.right_fit = 0

    def start2fit(self, image_data):
        """
        :param image_data: ImageLine Object
        :return:
        """
        histogram = np.sum(image_data.warped_binary[int(image_data.shape_w / 2):, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((image_data.warped_binary, image_data.warped_binary, image_data.warped_binary)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image_data.warped_binary.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image_data.warped_binary.nonzero()
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
            win_y_low = image_data.warped_binary.shape[0] - (window + 1) * window_height
            win_y_high = image_data.warped_binary.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin

            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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

        self.left_fit = left_fit
        self.right_fit = right_fit

    def fit(self):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

class ImageLine:
    def __init__(self, image, ret, mtx, dist, rvecs, tvecs):
        self.shape_h = image.shape[0]
        self.shape_w = image.shape[1]

        self.binary_output = np.zeros(shape=(self.shape_h,self.shape_w),dtype=np.float)
        self.binary_sobel = np.zeros(shape=(self.shape_h,self.shape_w),dtype=np.float)
        self.binary_hls_s = np.zeros(shape=(self.shape_h,self.shape_w),dtype=np.float)
        self.binary_mask = np.zeros(shape=(self.shape_h,self.shape_w),dtype=np.uint8)

        self.warped_binary = np.zeros_like(image)
        self.unwarped_binary = np.zeros_like(image)

        self.image = image
        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def to_bgr(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def undistort(self):
        self.image = cv2.undistort(self.to_bgr(), self.mtx, self.dist, None, self.mtx)

    def binary(self, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255), debug=False):
        # --------------------------- Binary Thresholding ----------------------------
        # Binary Thresholding is an intermediate step to improve lane line perception
        # it includes image transformation to gray scale to apply sobel transform and
        # binary slicing to output 0,1 type images according to pre-defined threshold.
        #
        # Also it's performed RGB to HSV transformation to get S information which in-
        # tensifies lane line detection.
        #
        # The output is a binary image combined with best of both S transform and mag-
        # nitude thresholding.

        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)

        # Sobel Transform
        sobelx = cv2.Sobel(hls[:, :, 1], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = 0  # cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_abs = np.abs(sobelx ** 2 + sobely ** 2)
        sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))

        self.binary_sobel[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1

        # Threshold color channel
        self.binary_hls_s[(hls[:, :, 2] >= s_thresh[0]) & (hls[:, :, 2] <= s_thresh[1])] = 1

        # Combine the two binary thresholds

        self.binary_output[(self.binary_hls_s == 1) | (self.binary_sobel == 1)] = 1
        self.binary_output = np.uint8(255 * self.binary_output / np.max(self.binary_output))

    def mask(self):
        # ---------------- MASKED IMAGE --------------------
        offset = 100
        mask_polyg = np.array([[(0 + offset, self.shape_h),
                                (self.shape_w / 2.5, self.shape_h / 1.65),
                                (self.shape_w / 1.8, self.shape_h / 1.65),
                                (self.shape_w, self.shape_h)]],
                              dtype=np.int)

        # This time we are defining a four sided polygon to mask
        # Applying polygon
        cv2.fillPoly(self.binary_mask, mask_polyg, 255)

        self.binary_mask = cv2.bitwise_and(self.binary_output, self.binary_mask)


    def warp(self, inverse_warp=False):

        line_dst_offset = 200

        src = [595, 452], \
              [685, 452], \
              [1110, self.shape_h], \
              [220, self.shape_h]

        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]

        src = np.float32([src])
        dst = np.float32([dst])

        if not inverse_warp:
            self.warped_binary = cv2.warpPerspective(self.binary_mask, cv2.getPerspectiveTransform(src, dst),
                                                     dsize=(self.shape_w,self.shape_h), flags=cv2.INTER_LINEAR)
        else:
            self.unwarped_binary = cv2.warpPerspective(self.binary_mask, cv2.getPerspectiveTransform(dst, src),
                                                       dsize=(self.shape_w, self.shape_h), flags=cv2.INTER_LINEAR)


