#%matplotlib inline
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc
import glob
import imageio
import sys
np.set_printoptions(threshold=sys.maxsize)

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only to get the nav_train
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select
    
#Threshold for the obstacles for RGB < 120 to identify obstacles
def color_thresh_obs(img, rgb_thresh=(120, 120, 120)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify yellow rock sample by thresholding betweeen lowest RGb for yellow color and RGB for the highest RGB for yellow color
def color_thresh_rock(img):
    # yellow_hsv = [30,255,255] # H is 0-179 degree not 360
    # convert RGB image to HSV image
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define lowerbound and upperbound for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    # detect color in image by masking pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(img, img, mask=mask)
    # convert result to binary
    binary_result = color_thresh(result, (0, 0, 0))
    # return binary result
    return binary_result

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 5
    source = np.float32([[14,140],
                        [300,140],
                        [200,95],
                        [120,95]])

    destination = np.float32([[Rover.img.shape[1] / 2 - dst_size, Rover.img.shape[0] - bottom_offset],
               [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
               [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
               [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])
    # 2) Apply perspective transform
    image_transform = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_train = color_thresh(image_transform,(160,160,160))
    #obs = 1 - nav_train 
    obs       = color_thresh_obs(image_transform,(120,120,120))#145,115,1
    rock      = color_thresh_rock(image_transform)

    #4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    navo = np.copy(nav_train)
    #for index,obj in enumerate(nav_train):
    #    #print("#"*90)
    #    # print(index)
    #    # print(obj[index])
    #    if index < 80:
    #        nav_train[index] = 0
    # nav_train[:, (nav_train.shape[1] // 2) + 7] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 8] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 9] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 10] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 11] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 12] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 13] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 14] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 15] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 16] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 17] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 18] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 19] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 20] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 21] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 22] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 23] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 24] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 25] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 26] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 27] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 28] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 29] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 30] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 31] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 32] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 33] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 34] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 35] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 36] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 37] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 38] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 39] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 40] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 41] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 42] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 43] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 44] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 45] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 46] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 47] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 48] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 49] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 50] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 51] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 52] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 53] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 54] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 55] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 56] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 57] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 58] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 59] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 60] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 61] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 62] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 63] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 64] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 65] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 66] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 67] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 68] *= 0
    nav_train[:, (nav_train.shape[1] // 2) + 69] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 70] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 71] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 72] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 73] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 74] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 75] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 76] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 77] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 78] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 79] *= 0
    # nav_train[:, (nav_train.shape[1] // 2) + 80] *= 0
    ########################################################################
    cv2.imshow("Super", nav_train * 255)

    Rover.vision_image[:,:,0] = obs * 255
    Rover.vision_image[:,:,1] = rock * 255
    Rover.vision_image[:,:,2] = nav_train * 255
    #Rover.vision_image[:, :, 2] = navo * 255
    # print("*"*80)
    # print(nav_train)
    # print("*"*80)

    threshed_navigable_crop = np.zeros_like(nav_train)
    threshed_obstacle_crop = np.zeros_like(obs)
    x1 = np.int(nav_train.shape[0]/2)      # index of start row  
    x2 = np.int(nav_train.shape[0])        # index of end row
    y1 = np.int(nav_train.shape[1]/3)      # index of start column
    y2 = np.int(nav_train.shape[1]*2/3)    # index of end column
    # crop from start to end row/column
    threshed_navigable_crop[x1:x2, y1:y2] = nav_train[x1:x2, y1:y2]
    threshed_obstacle_crop[x1:x2, y1:y2]  = obs[x1:x2, y1:y2]
    #
    # 5) Convert map image pixel values to rover-centric coords
    obs_roverx ,obs_rovery  = rover_coords(obs)
    rock_roverx,rock_rovery = rover_coords(rock)
    nav_train_roverx,nav_train_rovery = rover_coords(nav_train)
    
    xpix_nav_crop, ypix_nav_crop = rover_coords(threshed_navigable_crop)
    xpix_obs_crop, ypix_obs_crop = rover_coords(threshed_obstacle_crop)
      
    # 6) Convert rover-centric pixel values to world coordinates
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    obs_worldx,obs_worldy  = pix_to_world(xpix_obs_crop,  ypix_obs_crop ,xpos,ypos,Rover.yaw,Rover.worldmap.shape[0],10)
    
    rock_worldx,rock_worldy = pix_to_world(rock_roverx, rock_rovery,xpos,ypos,Rover.yaw,Rover.worldmap.shape[0],10)
    
    nav_train_worldx,nav_train_worldy = pix_to_world(xpix_nav_crop, ypix_nav_crop,xpos,ypos,Rover.yaw,Rover.worldmap.shape[0],10)

    #nav_train_worldx_fut, nav_train_worldy_fut =  pix_to_world(xpix_nav_crop, ypix_nav_crop,xpos + 50, ypos + 50,Rover.yaw,Rover.worldmap.shape[0],10)

    # print("*"*80)
    # print(Rover.worldmap[nav_train_worldy + 1 , nav_train_worldx + 1, 2])
    # print("+"*80)
    #if any(Rover.worldmap[nav_train_worldy_fut, nav_train_worldx_fut,2] > 0):
    # if (Rover.worldmap[nav_train_worldy + 1 , nav_train_worldx + 1, 2] > 0):####################################################################
    #     print("########################################################################")
    #     print("visited")
    #     Rover.mode = 'visited before'
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    if (np.float(np.abs(Rover.roll) % 360) <= 3) and (np.float(np.abs(Rover.pitch) % 360) <= 3):
        Rover.worldmap[obs_worldy, obs_worldx, 0] += 1
        Rover.worldmap[rock_worldy, rock_worldx, 1] += 1
        Rover.worldmap[nav_train_worldy, nav_train_worldx, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_anglesS
        # Set priority of picking up sample to be higher than navigating
        # if the sample is in vision, go pick it up first
    if (rock_roverx.any() or Rover.mode == 'goto_rock'):
        # Entering 'go-to-sample mode'
        if (Rover.mode != 'goto_rock'):
            Rover.mode = 'goto_rock'
        # if the sameple is in vision, set perception for navigating to the sample
        if (rock_roverx.any()):
            Rover.nav_dists, Rover.nav_angles = to_polar_coords(rock_roverx, rock_rovery)
            Rover.see_rock_error = 0
        # sometimes might mistakenly see the rock
        # Rover.see_rock_error is a commulative frame counter that rover might mistakenly see the sample
        else:
            Rover.see_rock_error += 1;
        # if mistakenly enter 'goto_rock' mode, and no longer see the sample, exit this mode
        if Rover.see_rock_error > 100:
            Rover.mode = 'stop'
    # if do not see any rock, set perception for a normal navigation
    # elif(nav_train_roverx.any() in Rover.visited_pos):
    #     print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
    #     print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
    #     print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
    #     print("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
    #     Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
    #elif(navx,nay is exists in visited)
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(
        #     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        # print(nav_train_roverx[0])
        #print(xpix_nav_crop[0])
        print(Rover.visited_pos[0][0])
    #elif any (obj[0] == 99.67 for obj in Rover.visited_pos):
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        # print(
        #     "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    else:
        # for x in Rover.visited_pos[1]:
        #     if x == nav_train_roverx.any():
        #         print(
        #             "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        #         print(
        #             "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        #         print(
        #             "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        #         print(
        #             "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(nav_train_roverx, nav_train_rovery)

    cv2.imshow("Rover Image",Rover.img)
    cv2.imshow("Rover Transform",image_transform)
    cv2.imshow("Nav_Train",nav_train * 255)
    cv2.imshow("Obstacles", obs * 255)
    cv2.imshow("Rocks", rock * 255)
    cv2.waitKey(1)

    return Rover
