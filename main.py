# Main file to import datasets, import parameters and initialize  Visual Odometry Pipeline 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import dataset_params
import re

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
    transformation_matrix, initial_landmarks = initialisation.init(params)

    #Call continuous operation pipeline


else:
    print("Program not yet defined for this dataset")