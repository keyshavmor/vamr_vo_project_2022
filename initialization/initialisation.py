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