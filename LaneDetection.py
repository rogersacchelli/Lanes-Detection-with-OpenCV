import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line:
    def __init__(self):
        self.right__detected = False
        self.left_inside_detected = False
        self.left_outside_detected = False
        self.right_inside_detected = False
        self.right_out_detected = False
        self.radius = 0.0
        self.turn_side = 0

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

