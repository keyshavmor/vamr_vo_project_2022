# Main file to import datasets, import parameters and initialize  Visual Odometry Pipeline 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import dataset_params
import re
import time

# Add path to initialisation code and dataset
sys.path.append('initialization')
import initialisation

sys.path.append('continuous_operation')
import continuous_operation

# We implement the visual odometry pipeline initially only for 3 datasets

num_of_datasets = 3
dataset_dictionary = {"parking":1, "KITTI":2, "Malaga":3}

user_dataset = input("Pick User Dataset (parking, KITTI, Malaga): ")

current_dataset = user_dataset

def convert_txt_to_array(path):

    if(path == 'initialization/test_dataset_parking/poses.txt'):
        matrix = np.zeros((1,12))
    else:
        matrix = np.zeros((1,3))

    # Open the text file
    with open(path, 'r') as file:
    
        # Read the contents of the file into a list of strings and convert them to a matrix
        data = file.readlines()
        for val in data:
            values = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', val)]
            array = np.array(values, dtype='float64')
            matrix = np.vstack((matrix, array))
        matrix = np.delete(matrix, 0, axis=0)

    return matrix

if(current_dataset == "parking"):

    # Obtain initial parameters, camera calibration and pose matrix to initialise pipeline

    params = dataset_params.parking_dataset_parameters()
    params["dataset"] = current_dataset

    K = convert_txt_to_array("initialization/test_dataset_parking/K.txt")
    params["K"] = K

    ground_truth_poses = convert_txt_to_array("initialization/test_dataset_parking/poses.txt")
    ground_truth_poses = ground_truth_poses[:, -9:]
    params["ground_truth_poses"] = ground_truth_poses

    #Initialise pipeline and obtain homography matrix among other values
    transformation_matrix, inlier_pts0, inlier_pts1, keypoints_0, keypoints_1, inliers, R, t = initialisation.init(params)

    parking_range = range(params["img_idx_1"], params["last_frame"]+1, 3)

    for i in parking_range:
    
        #Call continuous operation pipeline
        path = 'initialization/test_dataset_parking/img_'
        index = i
        string = str(index).zfill(5)
        path += string
        path += '.png'
        image = cv.imread(path)
        image_0 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow("Image", image_0)
        cv.waitKey(500)
        cv.destroyWindow("Image")
        
        path = 'initialization/test_dataset_parking/img_'
        index = i+3
        string = str(index).zfill(5)
        path += string
        path += '.png'
        image = cv.imread(path)
        image_1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imshow("Image", image_1)
        cv.destroyWindow("Image")
        cv.waitKey(500)
        time.sleep(0.01)
    #S_curr, transformation_matrix = processFrame(img_curr, img_prev, S_prev, params, transformation_matrix)


else:
    print("Program not yet defined for this dataset")