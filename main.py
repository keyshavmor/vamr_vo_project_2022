# Main file to import datasets, import parameters and initialize  Visual Odometry Pipeline 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import dataset_params

# We implement the visual odometry pipeline initially only for 3 datasets

num_of_datasets = 3
dataset_dictionary = {"parking":1, "KITTI":2, "Malaga":3}

user_dataset = input("Pick User Dataset (parking, KITTI, Malaga): ")

current_dataset = user_dataset

if(current_dataset == "parking"):

    params = dataset_params.parking_dataset_parameters()
    print(params)
else:
    print("Program not yet defined for this dataset")