# Main file to import datasets, import parameters and initialize  Visual Odometry Pipeline 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# We implement the visual odometry pipeline initially only for 3 datasets

num_of_datasets = 3
dataset_dictionary = {"parking":1, "KITTI":2, "Malaga":3}

