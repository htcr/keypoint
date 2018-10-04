import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    N = p1.shape[1]
    zeros = np.zeros((N, 3), dtype=np.float32)
    uv_homo = np.concatenate((p2.transpose((1, 0)), np.ones((N, 1), dtype=np.float32)), axis=1)
    
    sector1 = np.concatenate((uv_homo, zeros), axis=1).reshape(-1, 3)
    sector2 = np.concatenate((zeros, uv_homo), axis=1).reshape(-1, 3)
    
    xy_factor = p1.transpose((1, 0)).reshape(-1, 1)
    sector3 = -1.0*uv_homo.repeat(2, axis=0)*xy_factor

    A = np.concatenate((sector1, sector2, sector3), axis=1)
    U, S, V = np.linalg.svd(A)
    H2to1 = V[:, 8].reshape(3, 3)
    
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

