# File which will implement continuous operation for visual odometry

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def processFrame(image_1, image_0, inlier_pts0, keypoints_0, descriptor_0, inliers, R, t, params, transformation_matrix, initial_landmarks):
    
    # Create a SIFT descriptor object
    sift = cv.SIFT_create()

    # Compute feature vectors (descriptors) for the keypoints
    valid_kp_1, descriptor_1 = sift.detectAndCompute(image_1, None)

    # Draw the keypoints and their descriptors on the image
    img1_sift = cv.drawKeypoints(image_1, valid_kp_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the image
    cv.imshow("SIFT Descriptors", img1_sift)
    cv.waitKey(1000)
    cv.destroyWindow("SIFT Descriptors")

    # Match the keypoints between the two images using a brute-force matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_0, descriptor_1, k=2)

    # Use Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    print('Number of good keypoints:',len(good_matches))

    # Draw the matches on top of the images using the drawMatches function
    img_matches = cv.drawMatches(image_0, keypoints_0, image_1, valid_kp_1, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the resulting image
    #cv.imshow('Matched Keypoints', img_matches)
    #cv.waitKey(1000)
    #cv.destroyWindow('Matched Keypoints')

    pts_0 = np.float32([keypoints_0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_1 = np.float32([valid_kp_1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    K = params["K"]

    E, mask_e = cv.findEssentialMat(pts_0, pts_1, K, cv.RANSAC, 0.999, 1.0, 32000)

    ratio = np.count_nonzero(mask_e) / len(mask_e)
    print(f'Ratio of valid keypoints: {ratio}')

    # We select only inlier points
    inlier_pts0 = pts_0[mask_e.ravel()==1]
    inlier_pts1 = pts_1[mask_e.ravel()==1]

    inlier_pts0 = np.float64(np.squeeze(inlier_pts0))
    inlier_pts1 = np.float64(np.squeeze(inlier_pts1))

    # Compute the relative pose between the two cameras
    _, R, t, mask = cv.recoverPose(E, inlier_pts0, inlier_pts1)

    ratio = np.count_nonzero(mask) / len(mask)
    print(f'Ratio of valid keypoints: {ratio}')

    det = np.linalg.det(R)

    print('Determinant of Rotation Matrix:',det)

    M1 = np.float64(K @ np.eye(3,4))
    M2 = np.float64(K @ np.c_[R, t])

    points4D = cv.triangulatePoints(M1, M2, inlier_pts0.T, inlier_pts1.T)
    landmarks_1 = points4D[:3, :] / points4D[3, :]

    # Project the 3D points onto the images
    img_pts, _ = cv.projectPoints(landmarks_1, R, t, K, None)

    # Convert the projected points to integer coordinates
    img_pts = np.int32(img_pts).reshape(-1, 2)

    # Convert the inlier points to keypoint types
    kp1 = [cv.KeyPoint(x=p[0], y=p[1], size=20) for p in inlier_pts1]

    # Draw the keypoints and the projected points on the images
    image2 = cv.drawKeypoints(image_1, kp1, None)

    for pt in img_pts:
        cv.circle(image2, tuple(pt), 2, (0, 0, 255), -1)

    # Display the images
    cv.imshow('Image 2', image2)
    cv.waitKey(500)
    cv.destroyWindow('Image 2')

    return inlier_pts1, valid_kp_1, descriptor_1, mask_e, landmarks_1