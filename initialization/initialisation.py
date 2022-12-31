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

def detect_keypoints_descriptors(image_0, image_1):

    corner_points_0 = cv.cornerHarris(image_0, 2, 3, 0.04)
    threshold_factor_0 = (0.0000001)/(corner_points_0.max())
    # Threshold the corner points
    corner_points_0 = corner_points_0 > threshold_factor_0 * corner_points_0.max()

    # Convert the corner points to coordinates
    coords = np.column_stack(np.nonzero(corner_points_0))
    image = cv.cvtColor(image_0, cv.COLOR_GRAY2RGB)
    # Draw the corner points on the image
    for coord in coords:
        x, y = coord
        image = cv.circle(image, (y, x), 1, (0, 0, 255), 1) # Co-ordinates need to be inverted to fit the image
    # Display the image
    cv.imshow('image', image)
    cv.waitKey(1000)
    cv.destroyWindow('image')

    corner_points_1 = cv.cornerHarris(image_1, 2, 3, 0.04)
    threshold_factor_1 = (0.0000001)/(corner_points_1.max())
    # Threshold the corner points
    corner_points_1 = corner_points_1 > threshold_factor_1 * corner_points_1.max()
    # Convert the corner points to coordinates
    coords = np.column_stack(np.nonzero(corner_points_1))
    image = cv.cvtColor(image_1, cv.COLOR_GRAY2RGB)
    # Draw the corner points on the image
    for coord in coords:
        x, y = coord
        image = cv.circle(image, (y, x), 1, (0, 0, 255), 1) # Co-ordinates need to be inverted to fit the image
    # Display the image
    cv.imshow('image', image)
    cv.waitKey(1000)
    cv.destroyWindow('image')

    # Create a list of keypoints from the corner points
    keypoints_0 = []
    for y in range(corner_points_0.shape[0]):
        for x in range(corner_points_0.shape[1]):
            if corner_points_0[y, x] > 0:
                keypoint = cv.KeyPoint(x, y, size=10)
                keypoints_0.append(keypoint)

    # Create a SIFT descriptor object
    sift = cv.SIFT_create()

    # Compute feature vectors (descriptors) for the keypoints
    valid_kp_0, descriptor_0 = sift.compute(image_0, keypoints_0)

    # Create a list of keypoints from the corner points
    keypoints_1 = []
    for y in range(corner_points_1.shape[0]):
        for x in range(corner_points_1.shape[1]):
            if corner_points_1[y, x] > 0:
                keypoint = cv.KeyPoint(x, y, size=10)
                keypoints_1.append(keypoint)

    # Create a SIFT descriptor object
    sift = cv.SIFT_create()

    # Compute feature vectors (descriptors) for the keypoints
    valid_kp_1, descriptor_1 = sift.compute(image_1, keypoints_1)

    # Draw the keypoints and their descriptors on the image
    img0_sift = cv.drawKeypoints(image_0, valid_kp_0, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image
    cv.imshow("SIFT Descriptors", img0_sift)
    cv.waitKey(1000)
    cv.destroyWindow("SIFT Descriptors")

    # Draw the keypoints and their descriptors on the image
    img1_sift = cv.drawKeypoints(image_1, valid_kp_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image
    cv.imshow("SIFT Descriptors", img1_sift)
    cv.waitKey(1000)
    cv.destroyWindow("SIFT Descriptors")

    return valid_kp_0, valid_kp_1, img0_sift, img1_sift, descriptor_0, descriptor_1

def match_keypoints(image_0, image_1, valid_kp_0, valid_kp_1, descriptor_0, descriptor_1):

    # Match the keypoints between the two images using a brute-force matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_0, descriptor_1, k=2)

    # Use Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(len(good_matches))

    # Draw the matches on top of the images using the drawMatches function
    img_matches = cv.drawMatches(image_0, valid_kp_0, image_1, valid_kp_1, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting image
    cv.imshow('Matched Keypoints', img_matches)
    cv.waitKey(10000)
    cv.destroyWindow('Matched Keypoints')

    return good_matches

def compute_relative_pose(image_0, image_1, keypoints_0, keypoints_1, matched_kp, dataset_params):

    # Extract matched keypoints
    pts_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matched_kp]).reshape(-1, 1, 2)
    pts_1 = np.float32([keypoints_1[m.trainIdx].pt for m in matched_kp]).reshape(-1, 1, 2)

    # Estimate fundamental matrix using RANSAC
    F, mask = cv.findFundamentalMat(pts_0, pts_1, cv.FM_RANSAC, confidence=0.99, )

    # We select only inlier points
    inlier_pts0 = pts_0[mask.ravel()==1]
    inlier_pts1 = pts_1[mask.ravel()==1]

    K = dataset_params["K"]

    # Compute the relative pose between the two cameras
    _, R, t, mask = cv.recoverPose(F, inlier_pts0, inlier_pts1, K)



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
        image_0 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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
        image_1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow("Image", image_1)
        cv.waitKey(1000)
        cv.destroyWindow("Image")

    # Detect keypoints
    keypoints_0, keypoints_1, image_0, image_1, descriptor_0, descriptor_1 = detect_keypoints_descriptors(image_0, image_1)

    # Match Keypoints
    matched_kp = match_keypoints(image_0, image_1, keypoints_0, keypoints_1, descriptor_0, descriptor_1)

    # Compute relative pose of second image
    orientation, localization, inliers = compute_relative_pose(image_0, image_1, keypoints_0, keypoints_1, matched_kp, dataset_params)

    # Triangulate landmarks from matched keypoints
    #transformation_matrix, initial_landmarks = triangulate(matched_kp_0, matched_kp_1, inliers, orientation, localization, dataset_params)

    #Visualise matched features and inliers

    # Use cv2.drawMatches to display the matched keypoints
    #matched_features = cv2.drawMatches(image_0, matched_kp_0, image_1, matched_kp_1, None)

    # Display the output image
    #cv2.imshow("Matched keypoints", matched_features)
    #cv2.waitKey(1000)
    #cv2.destroyWindow("Matched keypoints")

    # Use cv2.drawMatches to display the inlier matched keypoints
    #matched_inliers = cv2.drawMatches(image_0, matched_kp_0, image_1, matched_kp_1, inliers, None)

    # Display the output image
    #cv2.imshow("Matched inliers", matched_inliers)
    #cv2.waitKey(1000)
    #cv2.destroyWindow("Matched inliers")