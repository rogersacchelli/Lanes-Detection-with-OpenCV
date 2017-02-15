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
from LaneDetection import Line, ImageLine

out_examples = False
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

    lines = Line()
    image_data = ImageLine(np.zeros(shape=(720,1280,3),dtype=np.float32), ret, mtx, dist, rvecs, tvecs)

    for i, img in enumerate(vid):
        t0 = time.time()
        image_data.image = img
        image_data.undistort()
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


        image_data.binary()
        image_data.mask()
        image_data.warp()

        # if out_examples:
        #     # Count from mid frame beyond
        #     histogram = np.sum(image_data.warped_binary[int(image_data.warped_binary.shape[0] / 2):, :], axis=0)
        #     plt.plot(histogram)
        #     plt.savefig('histogram.jpg')
        #     plt.close()
        #
        #     plt.figure(figsize=(21, 15))
        #     for i, img in enumerate([img, img_b, img_w, imread('histogram.jpg')]):
        #         plt.subplot(2, 2, i + 1)
        #         plt.imshow(img, cmap='gray')
        #         if i == 3:
        #             plt.axis('off')
        #     plt.show()

        if not lines.right__detected or not lines.left_detected:
            lines.start2fit(image_data)
        else:
            lines.fit(image_data)

        try:
            mov_avg_left = np.append(mov_avg_left,np.array([left_fit]), axis=0)
            mov_avg_right = np.append(mov_avg_right,np.array([right_fit]), axis=0)

        except:

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

        if abs(left_fit[0]) < 5e-5:
            lines.turn_side = 0
        elif left_fit[0] > 0:
            lines.turn_side = 1
        else:
            lines.turn_side = -1


        t_draw0 = time.time()
        #final = draw_lines(img, img_w, left_fit, right_fit, perspective=[src,dst])
        t_draw = time.time() - t_draw0

        # print('fps: %d' % int((1./(t1-t0))))
        #print('undist: %f [ms] | bin: %f [ms]| warp: %f [ms]| fit: %f [ms]| draw: %f [ms] | fps %f'
        #      % (t_dist * 1000, t_bin * 1000, t_warp * 1000, t_fit * 1000, t_draw * 1000, 1./(time.time() - t_fps0)))
        print(lines.turn_side)
        #cv2.imshow('final', final)

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
    #newwarp = warp(color_warp, perspective[1], perspective[0])
    # Combine the result with the original image
    #result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
    cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
    #newwarp_lines = warp(color_warp_lines, perspective[1], perspective[0])

    #result = cv2.addWeighted(img, 1, newwarp_lines, 1, 0)

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
    #    cv2.putText(result, line, (0,i), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
    #return result



if __name__ == '__main__':
    main()
