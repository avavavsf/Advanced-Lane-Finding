#import the needed modules
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from moviepy.editor import VideoFileClip

# define camera related parameters and functions
class Camera():


    def __init__(self):
        self.width = 1280
        self.height = 720
        self.calibration_images = 'camera_cal/cal*.jpg'
        self.calibration_parameters = 'temp/cal_pickle.p'
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
         # Perspective Transform matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        # Inverse Perspective Transform matrix
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
    
    # calculate and save the camera matrix and undistortion coefficients to pickle file
    def undistort_parameters(self):
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)    

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.    

        # Make a list of calibration images
        images = glob.glob(self.calibration_images)    

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)    

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        #calculate the camera matrix and undistortion coefficients
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open(self.calibration_parameters, "wb" ) )    
    
    
    #undistorted a image using the camera matrix and undistortion coeffi.
    def undistort_image(self,img):
        #load the saved camera matrix and distortion coefficients
        dist_pickle = pickle.load( open(self.calibration_parameters, "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        #conduct the undistortion
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)    

        return undistorted    

    # methods to extract likely lanes
    # directional gradient
    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return the result
        return binary_output    

    # magnitude of the gradient
    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # Return the binary image
        return binary_output    

    # direction of the gradient
    def dir_threshold(self,image, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # Return the binary image
        return binary_output    

    #the S channel threshold from HSL image repersentatives
    def s_channel(self,img,thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return s_binary    

    # select combine the above methods to get a binary image
    def get_binary_img(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        #the Threshold x gradient
        sxbinary = self.abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
        #the S channel threshold
        s_binary = self.s_channel(img,thresh=s_thresh)             
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary    

    #perspective transform to have a bird view of the lane
    def pers_trans(self, img,reverse=False):
        img_size = (img.shape[1], img.shape[0])
        if reverse:
            warped = cv2.warpPerspective(img, self.Minv, img_size, flags=cv2.INTER_NEAREST)
        else:
            warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_NEAREST)
        return warped

class Lines(Camera):
    def __init__(self):
        Camera.__init__(self)     
        #polynomial coefficients averaged over the last n iterations
        self.left_best_fit = None  
        #polynomial coefficients for the most recent fit
        self.left_pre_fitx = None
        self.right_best_fit = None  
        #polynomial coefficients for the most recent fit
        self.right_pre_fitx = None
        #frame number
        self.frame_number = 0
        #number of frame of continous undetected
        self.n_undetected = 0

    
    # calculate the curvature, and the offset from lane center
    def get_curvature_offsets(self,img,leftx,lefty,rightx,righty):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30./(self.dst[1][1]-self.dst[0][1]) # meters per pixel in y dimension
        xm_per_pix = 3.7/(self.dst[2][0]-self.dst[1][0]) # meters per pixel in x dimension    

        #calculate the curvature and offsets
        y_eval = img.shape[0]
        center_x = (img.shape[1])/2*xm_per_pix
        center_y = img.shape[1]*ym_per_pix    

        # Create an image to draw the lines on
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        #curvature
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        #offsets
        leftx_lane = left_fit[0]*center_y**2 + left_fit[1]*center_y + left_fit[2]
        rightx_lane = right_fit[0]*center_y**2 + right_fit[1]*center_y + left_fit[2]
        offsets = leftx_lane + rightx_lane - center_x
        return left_curverad,right_curverad,offsets    

    #implement Sliding Windows and Fit a Polynomial
    def init_lane_locate(self,binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
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
        right_lane_inds = []        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
     
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]  

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return leftx,lefty,rightx,righty,left_fit, right_fit        

    # Assume you now have a new warped binary image 
    # from the next frame of video,now we fit the new frame
    def lane_locate(self,binary_warped,left_fit,right_fit):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))          # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return leftx,lefty,rightx,righty    

    # Evaluates polynomial and finds value at given point
    def get_x_for_line(self,line_fit, line_y):
        poly = np.poly1d(line_fit)
        return poly(line_y)     

    def lane_fit(self,img,left_fit,right_fit):   
        # Recast the x and y points into usable format for cv2.fillPoly()
        y_vals = np.arange(0, img.shape[0])
        left_fitx = self.get_x_for_line(left_fit, y_vals)
        right_fitx = self.get_x_for_line(right_fit, y_vals)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, y_vals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_vals])))])
        pts = np.hstack((pts_left, pts_right))  
        return pts  

    def if_detected(self,frame_number,curvature,cur_fitx,pre_fitx):
        d = True
        if curvature<587: 
            #print("Bad lane detection because of wrong curvature at frame {}".format(frame_number))
            d = False   
    
        if abs(cur_fitx - pre_fitx)>15:
            #print("Bad lane detection because of too different at frame {}".format(frame_number))
            d = False   

        return d    

    def lane_drawing(self,img):  
        camera = Camera()
        #read and undistort the image
        img = camera.undistort_image(img)
        ##apply threshold to get the binary image
        binary_img = camera.get_binary_img(img, (170, 255), (20, 100))
        #apply the perspective transform
        binary_warped = camera.pers_trans(binary_img)   

        #first frame
        if self.frame_number == 0:
            leftx,lefty,rightx,righty,self.left_best_fit,self.right_best_fit = self.init_lane_locate(binary_warped)
            #calculate the curvature and offsets
            left_curvature,right_curvature,offsets = self.get_curvature_offsets(img,leftx,lefty,rightx,righty)
            #get the bottom x 
            self.left_pre_fitx = self.get_x_for_line(self.left_best_fit, self.height)
            self.right_pre_fitx = self.get_x_for_line(self.right_best_fit, self.height)
        else:
            leftx,lefty,rightx,righty = self.lane_locate(binary_warped,self.left_best_fit,self.right_best_fit)       

            #fit the lanes
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            #calculate the curvature and offsets
            left_curvature,right_curvature,offsets = self.get_curvature_offsets(img,leftx,lefty,rightx,righty)
            #get the bottom x 
            left_fitx = self.get_x_for_line(left_fit, self.height)
            right_fitx = self.get_x_for_line(right_fit, self.height)
            #test if the detected lane is resonable
            left_detected = self.if_detected(self.frame_number,left_curvature,left_fitx,self.left_pre_fitx)
            right_detected = self.if_detected(self.frame_number,right_curvature,right_fitx,self.right_pre_fitx)
            if left_detected and right_detected:
                self.left_pre_fitx = left_fitx
                self.right_pre_fitx = right_fitx
                self.left_best_fit = self.left_best_fit * 0.8 + left_fit * 0.2 
                self.right_best_fit = self.right_best_fit * 0.8 + right_fit * 0.2
                self.n_undetected = 0
            else:
                n_undetected = self.n_undetected + 1   

            if self.n_undetected > 5:
                print('init lane locate')
                _,_,_,_,self.left_best_fit,self.right_best_fit = self.init_lane_locate(binary_warped)
                #get the bottom x 
                self.left_pre_fitx = self.get_x_for_line(self.left_best_fit, self.height)
                self.right_pre_fitx = self.get_x_for_line(self.right_best_fit, self.height)    

        
        #lane fitting
        pts = self.lane_fit(img,self.left_best_fit,self.right_best_fit)
        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))       
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = camera.pers_trans(color_warp,reverse=True)    
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, .3, 0)
            

        #write the curvature and offsets
        if offsets < 0:
            cv2.putText(result, 'Vehicle is {:.2f} meters left of center'.format(offsets),(20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(result, 'Vehicle is {:.2f} meters right of center'.format(offsets),(20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'Radius of curvature is {} meters'.format(int((left_curvature + right_curvature)/2)),(20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        self.frame_number = self.frame_number + 1
        
        return result   

    
