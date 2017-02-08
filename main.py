import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize, imsave

out_examples = 0


def main():

    # ------------------------ Camera Calibration ------------------------
    # As calibration may take some time, save calibration data into pickle file to speed up testing
    if not os.path.exists('calibration.p'):
        # Read all jpg files from calibration image folder
        images = glob.glob('camera_cal/*.jpg')

        with open('calibration.p', mode='wb') as f:
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(images, nx=9, ny=6)
            pickle.dump([ret, mtx, dist, rvecs, tvecs], f)
            f.close()
    else:
        with open('calibration.p', mode='rb') as f:
            ret, mtx, dist, rvecs, tvecs = pickle.load(f)
            f.close()

    if out_examples:
        # output undistorted image to output_image
        to_calibrate = imread('camera_cal/calibration2.jpg')
        imsave('output_images/calibration2_calibrated.jpg', cv2.undistort(to_calibrate, mtx, dist, None, mtx))

    cap = cv2.VideoCapture('project_video.mp4')

    while (cap.isOpened()):

        ret, img = cap.read()

        img = cv2.undistort(img, mtx, dist, None, mtx)

        # --------------------------- Binary Thresholding ----------------------------

        if out_examples:
            test_images = glob.glob('test_images/*.jpg')
            plt.figure(figsize=(14, 10))
            for i, img in enumerate(test_images):
                img_b = image_binary(cv2.undistort(cv2.imread(img), mtx, dist, None, mtx))
                plt.subplot(3, 3, i + 1)
                plt.axis('off')
                plt.title('%s' % str(img))
                plt.imshow(img_b, cmap='gray')
            plt.show()

        img_b = image_binary(img)

        # ---------------------------- Perspective Transform --------------------------

        img_w = warp(img_b)

        if out_examples:
            # Count from mid frame beyond
            histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
            plt.plot(histogram)
            plt.savefig('histogram.jpg')
            plt.close()

            # plt.figure(figsize=(21, 15))
            # for i, img in enumerate([img, img_b, img_w, imread('histogram.jpg')]):
            #     plt.subplot(2, 2, i + 1)
            #     plt.imshow(img, cmap='gray')
            #     if i == 3:
            #         plt.axis('off')
            # plt.show()

        left_fit, right_fit = sliding_windown(img_w)

        final = draw_lines(img, img_w, left_fit, right_fit, text=radius_curvature(left_fit,right_fit, img_w.shape[0]))

        cv2.imshow('final', final)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

def calibrate_camera(image_files, nx, ny):
    objpoints = []
    imgpoints = []

    objp = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for i in image_files:
        img = cv2.imread(i)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def image_binary(img, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255)):

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]


    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = 0 #cv2.Sobel(hsv[:, :, 2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.abs(sobelx**2 + sobely**2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))

    sobel_binary = np.zeros_like(gray)
    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1

    # Threshold color channel

    s_binary = np.zeros_like(hls[:, :, 2])
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1

    #plt.imshow(combined_binary, cmap='gray')
    #plt.show()

    return combined_binary


def warp(img):

    img_size = img.shape

    src_a = (585, 457)
    src_b = (700, 457)
    src_c = (1110, img_size[0])
    src_d = (220, img_size[0])

    dst_a = (src_d[0]+100, 0)
    dst_b = (src_c[0]-100, 0)
    dst_c = (src_c[0]-100,src_c[1])
    dst_d = (src_d[0]+100,src_d[1])

    src = np.float32([[src_a[0], src_a[1]], [src_b[0], src_b[1]], [src_c[0], src_c[1]], [src_d[0], src_d[1]]])
    dst = np.float32([[dst_a[0], dst_a[1]], [dst_b[0], dst_b[1]], [dst_c[0], dst_c[1]], [dst_d[0], dst_d[1]]])
    
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                               dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)


def unwarp(img):
    img_size = img.shape

    src_a = (585, 457)
    src_b = (700, 457)
    src_c = (1110, img_size[0])
    src_d = (220, img_size[0])

    dst_a = (src_d[0] + 100, 0)
    dst_b = (src_c[0] - 100, 0)
    dst_c = (src_c[0] - 100, src_c[1])
    dst_d = (src_d[0] + 100, src_d[1])

    src = np.float32([[src_a[0], src_a[1]], [src_b[0], src_b[1]], [src_c[0], src_c[1]], [src_d[0], src_d[1]]])
    dst = np.float32([[dst_a[0], dst_a[1]], [dst_b[0], dst_b[1]], [dst_c[0], dst_c[1]], [dst_d[0], dst_d[1]]])

    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(dst, src),
                               dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)


def sliding_windown(img_w):
    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_w, img_w, img_w)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

    # Generate x and y values for plotting
    # ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    return left_fit, right_fit


def fit_from_lines(left_fit, right_fit, img):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
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


def radius_curvature(left_fit, right_fit, img_height=720.):

    y_eval = img_height

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return (left_curverad, right_curverad)

def draw_lines(img, img_w, left_fit, right_fit, text):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    src_a = (585, 457)
    src_b = (700, 457)
    src_c = (1110, img.shape[0])
    src_d = (220, img.shape[0])

    dst_a = (src_d[0] + 100, 0)
    dst_b = (src_c[0] - 100, 0)
    dst_c = (src_c[0] - 100, src_c[1])
    dst_d = (src_d[0] + 100, src_d[1])

    src = np.float32([[src_a[0], src_a[1]], [src_b[0], src_b[1]], [src_c[0], src_c[1]], [src_d[0], src_d[1]]])
    dst = np.float32([[dst_a[0], dst_a[1]], [dst_b[0], dst_b[1]], [dst_c[0], dst_c[1]], [dst_d[0], dst_d[1]]])

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, cv2.getPerspectiveTransform(dst, src), (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    text = "radius = %s [m]" % str(round((float(text[0]) + float(text[1]))/2.,2))
    cv2.putText(result, str(text), (0,50), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    return result

if __name__ == '__main__':
    main()
