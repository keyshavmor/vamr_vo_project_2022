# Initialisation file in Python to obtain 2 initial frames, bootstrap their poses

# The initialization file takes an input of two images (keyframes) which are then used to
# obtain keypoints on image plane and corresponding 3D keypoints in world co-ordinate system. 
# The transformation matrix to convert image points to 3D world coordinate system is also computed.
#
#   INPUT:
#       img_0                 : first image of data set
#       img_n                 : next image (nth image of dataset separated by a sufficient baseline)
#       camera_intrinsics     : Camera calibration matrix K and initial pose matrix P.
#
#   OUTPUT:
#       transformation_matrix : Dimension - 4x4, transform from image frame to 3D world co-ordination system
#       initial_keypoints     : Dimension - Nx2, N is number of keypoints. Initial keypoints in the image plane                   
#       initial_landmarks     : Dimension - Nx2, N is number of landmarks. initial landmarks which where triangulated from intial keypoints.

# Initialisation algorithm

# 1. Detect keypoints from the two input images and camera parameters of the initial frame
# 2. Obtain Keypoint descriptors
# 3. Match keypoints in the two images
# 4. Compute relative pose of second image with previous image
# 5. Traingulate landmarks in the images
# 6. Plot matched features and keypoints

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Detect keypoints in the two images
def init(dataset_params):

    if(dataset_params["dataset"] == "parking"):
        path = dataset_params["path"]
        path += '/img_'

        index = dataset_params["img_idx_0"]
        final_img_index_0 = 0 + index
        string = str(final_img_index_0).zfill(5)
        path += string
        path += '.png'
    
        # Obtain Keyframe 0
        image = cv.imread(path)
        image_0 = image.astype(np.uint8)
        cv.imshow("Image", image_0)
        cv.waitKey(1000)
        cv.destroyWindow("Image")

        path = dataset_params["path"]
        path += '/img_'
        index = dataset_params["img_idx_1"]
        final_img_index_1 = 0 + index
        string = str(final_img_index_1).zfill(5)
        path += string
        path += '.png'

        # Obtain Keyframe 1
        image = cv.imread(path)
        image_1 = image.astype(np.uint8)
        cv.imshow("Image", image_1)
        cv.waitKey(1000)
        cv.destroyWindow("Image")

        