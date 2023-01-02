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
from mpl_toolkits.mplot3d import Axes3D

def detect_keypoints_descriptors(image_0, image_1):

    # Create a SIFT descriptor object
    sift = cv.SIFT_create()

    # Compute feature vectors (descriptors) for the keypoints
    valid_kp_0, descriptor_0 = sift.detectAndCompute(image_0, None)

    # Create a SIFT descriptor object
    sift = cv.SIFT_create()

    # Compute feature vectors (descriptors) for the keypoints
    valid_kp_1, descriptor_1 = sift.detectAndCompute(image_1, None)

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

    return valid_kp_0, valid_kp_1, descriptor_0, descriptor_1

def match_keypoints(image_0, image_1, valid_kp_0, valid_kp_1, descriptor_0, descriptor_1):

    # Match the keypoints between the two images using a brute-force matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_0, descriptor_1, k=2)

    # Use Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    print(len(good_matches))

    # Draw the matches on top of the images using the drawMatches function
    img_matches = cv.drawMatches(image_0, valid_kp_0, image_1, valid_kp_1, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting image
    cv.imshow('Matched Keypoints', img_matches)
    cv.waitKey(1000)
    cv.destroyWindow('Matched Keypoints')

    return good_matches

def compute_relative_pose(image_0, image_1, keypoints_0, keypoints_1, matched_kp, dataset_params):

    # Extract matched keypoints
    pts_0 = np.float32([keypoints_0[m.queryIdx].pt for m in matched_kp]).reshape(-1, 1, 2)
    pts_1 = np.float32([keypoints_1[m.trainIdx].pt for m in matched_kp]).reshape(-1, 1, 2)

    K = dataset_params["K"]

    E, mask_e = cv.findEssentialMat(pts_0, pts_1, K, cv.RANSAC, 0.999, 1.0, 32000)

    ratio = np.count_nonzero(mask_e) / len(mask_e)
    print(f'Ratio of valid keypoints: {ratio}')

    # We select only inlier points
    inlier_pts0 = pts_0[mask_e.ravel()==1]
    inlier_pts1 = pts_1[mask_e.ravel()==1]

    # Compute the relative pose between the two cameras
    _, R, t, mask = cv.recoverPose(E, inlier_pts0, inlier_pts1)

    ratio = np.count_nonzero(mask) / len(mask)
    print(f'Ratio of valid keypoints: {ratio}')

    return R, t, mask_e, inlier_pts0, inlier_pts1

def triangulate(matched_kp, inlier_pts0, inlier_pts1, keypoints_0, keypoints_1, inliers, R, t, dataset_params, image_0, image_1):
    # Compute the projection matrices.
    
    K = dataset_params["K"]
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = np.dot(K,P1)
    T = np.hstack((R,t))
    T - np.dot(K,T)

    # Triangulate the points.
    points4D = cv.triangulatePoints(P1, T, inlier_pts0.T, inlier_pts1.T)

    points3D = cv.convertPointsFromHomogeneous(points4D.T)

    #squeeze_3d = np.squeeze(points3D)

    # Project the 3D points onto the images
    #img_pts, _ = cv.projectPoints(points3D, R, t, K, None)

    # Convert the projected points to integer coordinates
    #img_pts = np.int32(img_pts).reshape(-1, 2)


    # Draw the keypoints and the projected points on the images
    #image1 = cv.drawKeypoints(image_0, keypoints_0, None)
    #image2 = cv.drawKeypoints(image_1, keypoints_1, None)

    #for pt in img_pts:
    #    cv.circle(image1, tuple(pt), 5, (0, 255, 0), -1)
    #    cv.circle(image2, tuple(pt), 5, (0, 255, 0), -1)

    # Display the images
    #cv.imshow('Image 1', image1)
    #cv.imshow('Image 2', image2)
    #cv.waitKey(30000)
    #cv.destroyWindow('Image 1')
    #cv.destroyWindow('Image 2')


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
    keypoints_0, keypoints_1, descriptor_0, descriptor_1 = detect_keypoints_descriptors(image_0, image_1)

    # Match Keypoints
    matched_kp = match_keypoints(image_0, image_1, keypoints_0, keypoints_1, descriptor_0, descriptor_1)

    # Compute relative pose of second image
    R, t, inliers, inlier_pts0, inlier_pts1 = compute_relative_pose(image_0, image_1, keypoints_0, keypoints_1, matched_kp, dataset_params)

    # Triangulate landmarks from matched keypoints
    transformation_matrix, initial_landmarks = triangulate(matched_kp, inlier_pts0, inlier_pts1, keypoints_0, keypoints_1, inliers, R, t, dataset_params, image_0, image_1)

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