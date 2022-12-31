# File to export dataset parameters for initialising Visual Odometry Pipeline

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def parking_dataset_parameters():

    parking_params = {}

    # Parameters for bootstrapped initialisation
    path_data = "initialization/test_dataset_parking"

    parking_params["path"] = path_data

    # Frame selection
    m = 0 # Keyframe image 1
    n = 3 # Keyframe image 3

    parking_params["img_idx_0"] = m
    parking_params["img_idx_1"] = n

    # Harris parameters
    corner_patch_size = 9
    kappa = 0.08
    num_of_keypoints = 6000
    non_maxima_supression_radius = 20

    parking_params["patch_size"] = corner_patch_size
    parking_params["kappa"] = kappa
    parking_params["num_keypoints"] = num_of_keypoints
    parking_params["non_max_radius"] = non_maxima_supression_radius

    # Descriptor parameters to describe keypoint neighbourhood
    descriptor_radius = 21

    parking_params["descriptor_radius"] = descriptor_radius

    # Match keypoints
    matching_mode = 'klt' # {'klt','patch_matching'}

    parking_params["matching_alg"] = matching_mode

    # Patch matching parameters parking
    patch_lambda = 8 # Treshold for descriptors to match

    parking_params["lambda"] = patch_lambda

    # KLT parameters
    max_bilinear_error_klt = 0.8 #The forward-backward error treshold

    parking_params["max_klt_error"] = max_bilinear_error_klt

    num_pyramid_levels = 6 #The number of pyramid level

    parking_params["num_klt_levels"] = num_pyramid_levels

    klt_block_size = 21 #Size of neighborhood around each point being tracked

    parking_params["klt_block_size"] = klt_block_size

    max_klt_iteration = 40 #Maximum number of search iterations for each point

    parking_params["max_klt_iterations"] = max_klt_iteration

    return parking_params