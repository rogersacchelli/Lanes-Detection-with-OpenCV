import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import pylab
import time
import imageio
from scipy.misc import imread, imresize, imsave

out_examples = 0
MOV_AVG_LENGTH = 5


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
        to_calibrate = imread('camera_cal/calibration3.jpg')
        imsave('output_images/calibration3_calibrated.jpg', cv2.undistort(to_calibrate, mtx, dist, None, mtx))

    vid = imageio.get_reader('project_video.mp4', 'ffmpeg')

    for i, img in enumerate(vid):

        t_dist0 = time.time()
        t_fps0 = t_dist0
        img = cv2.undistort(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mtx, dist, None, mtx)
        t_dist = time.time() - t_dist0


        # --------------------------- Binary Thresholding ----------------------------
        #
        # if out_examples:
        #     test_images = glob.glob('test_images/*.jpg')
        #     plt.figure(figsize=(14, 10))
        #     for i, img in enumerate(test_images):
        #         img_b = image_binary(cv2.undistort(cv2.imread(img), mtx, dist, None, mtx))
        #         plt.subplot(3, 3, i + 1)
        #         plt.axis('off')
        #         plt.title('%s' % str(img))
        #         plt.imshow(img_b, cmap='gray')
        #     plt.show()

        t_bin0 = time.time()
        img_b = image_binary(img)
        t_bin = time.time() - t_bin0

        # ---------------------------- Perspective Transform --------------------------

        t_warp0 = time.time()
        #src = [585, 457], [700, 457], [1110, img_b.shape[0]], [220, img_b.shape[0]]

        line_dst_offset = 200
        src = [595, 452], \
              [685, 452], \
              [1110, img_b.shape[0]], \
              [220, img_b.shape[0]]

        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]

        img_w = warp(img_b, src, dst)
        t_warp = time.time() - t_warp0

        if out_examples:
            # Count from mid frame beyond
            histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
            plt.plot(histogram)
            plt.savefig('histogram.jpg')
            plt.close()

            plt.figure(figsize=(21, 15))
            for i, img in enumerate([img, img_b, img_w, imread('histogram.jpg')]):
                plt.subplot(2, 2, i + 1)
                plt.imshow(img, cmap='gray')
                if i == 3:
                    plt.axis('off')
            plt.show()


        t_fit0 = time.time()
        try:
            left_fit, right_fit = fit_from_lines(left_fit, right_fit, img_w)

            mov_avg_left = np.append(mov_avg_left,np.array([left_fit]), axis=0)
            mov_avg_right = np.append(mov_avg_right,np.array([right_fit]), axis=0)

        except:
            left_fit, right_fit = sliding_windown(img_w)

            mov_avg_left = np.array([left_fit])
            mov_avg_right = np.array([right_fit])

        left_fit = np.array([np.mean(mov_avg_left[::-1][:,0][0:MOV_AVG_LENGTH]),
                             np.mean(mov_avg_left[::-1][:,1][0:MOV_AVG_LENGTH]),
                             np.mean(mov_avg_left[::-1][:,2][0:MOV_AVG_LENGTH])])
        right_fit = np.array([np.mean(mov_avg_right[::-1][:,0][0:MOV_AVG_LENGTH]),
                             np.mean(mov_avg_right[::-1][:,1][0:MOV_AVG_LENGTH]),
                             np.mean(mov_avg_right[::-1][:,2][0:MOV_AVG_LENGTH])])

        if mov_avg_left.shape[0] > 1000:
            mov_avg_left = mov_avg_left[0:MOV_AVG_LENGTH]
        if mov_avg_right.shape[0] > 1000:
            mov_avg_right = mov_avg_right[0:MOV_AVG_LENGTH]


        t_fit = time.time() - t_fit0

        t_draw0 = time.time()
        final = draw_lines(img, img_w, left_fit, right_fit, perspective=[src,dst])
        t_draw = time.time() - t_draw0

        # print('fps: %d' % int((1./(t1-t0))))
        print('undist: %f [ms] | bin: %f [ms]| warp: %f [ms]| fit: %f [ms]| draw: %f [ms] | fps %f'
              % (t_dist * 1000, t_bin * 1000, t_warp * 1000, t_fit * 1000, t_draw * 1000, 1./(time.time() - t_fps0)))
        cv2.imshow('final', final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def calibrate_camera(image_files, nx, ny):
    objpoints = []
    imgpoints = []

    objp = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for i in image_files:
        img = cv2.imread(i)
        if img.shape[0] != 720:
            img = cv2.resize(img,(1280, 720))
        cv2.imshow('image',img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

        if ret:
            print("Calibrated!")
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

    #plt.imshow(combined_binary, cmap='gray')
    #plt.show()

    # ---------------- MASKED IMAGE --------------------
    offset = 100
    mask_polyg = np.array([[(0 + offset, img.shape[0]),
                            (img.shape[1] / 2.5, img.shape[0] / 1.65),
                            (img.shape[1] / 1.8, img.shape[0] / 1.65),
                            (img.shape[1], img.shape[0])]],
                          dtype=np.int)

    # mask_polyg = np.concatenate((mask_polyg, mask_polyg, mask_polyg))

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask_img = np.zeros_like(combined_binary)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    # Applying polygon
    cv2.fillPoly(mask_img, mask_polyg, ignore_mask_color)
    masked_edges = cv2.bitwise_and(combined_binary, mask_img)

    return masked_edges


def warp(img, src, dst):

    src = np.float32([src])
    dst = np.float32([dst])
    
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
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


def fit_from_lines(left_fit, right_fit, img_w):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img_w.nonzero()
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


def draw_lines(img, img_w, left_fit, right_fit, perspective):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #color_warp_center = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, perspective[1], perspective[0])
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
    cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
    newwarp_lines = warp(color_warp_lines, perspective[1], perspective[0])

    result = cv2.addWeighted(result, 1, newwarp_lines, 1, 0)

    # ----- Radius Calculation ------ #

    img_height = img.shape[0]
    y_eval = img_height

    ym_per_pix = 30 / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

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

    # ----- Off Center Calculation ------ #

    lane_width = (right_fit[2] - left_fit[2]) * xm_per_pix
    center = (right_fit[2] - left_fit[2]) / 2
    off_left = (center - left_fit[2]) * xm_per_pix
    off_right = -(right_fit[2] - center) * xm_per_pix
    off_center = round((center - img.shape[0] / 2.) * xm_per_pix,2)

    # --- Print text on screen ------ #
    #if radius < 5000.0:
    text = "radius = %s [m]\noffcenter = %s [m]" % (str(radius), str(off_center))
    #text = "radius = -- [m]\noffcenter = %s [m]" % (str(off_center))

    for i, line in enumerate(text.split('\n')):
        i = 50 + 20 * i
        cv2.putText(result, line, (0,i), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    return result



if __name__ == '__main__':
    main()
