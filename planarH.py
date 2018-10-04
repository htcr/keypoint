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
    U, S, Vt = np.linalg.svd(A)
    H2to1 = Vt[8, :].reshape(3, 3)

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
    max_inlier_num = -1
    inlier_p1s = None
    inlier_p2s = None
    ransac_best_H = None
    all_p1s = locs1[matches[:, 0], 0:2].transpose((1, 0))
    all_p2s = locs2[matches[:, 1], 0:2].transpose((1, 0))
    for i in range(num_iter):
        # sample minimum set
        mini_matches_id = np.random.choice(np.arange(matches.shape[0]), 4)
        p1s = all_p1s[:, mini_matches_id]
        p2s = all_p2s[:, mini_matches_id]
        
        curH2to1 = computeH(p1s, p2s)
        
        all_p2to1s = curH2to1 @ np.concatenate((all_p2s, np.ones((1, all_p2s.shape[1]), dtype=np.float32)), axis=0)
        all_p2to1s[0, :] /= all_p2to1s[2, :]
        all_p2to1s[1, :] /= all_p2to1s[2, :]
        all_p2to1s = all_p2to1s[0:2, :]

        dists = np.sum((all_p2to1s - all_p1s)**2, axis=0)**0.5
        inlier_flags = dists < tol
        cur_inlier_num = np.sum(inlier_flags)
        if cur_inlier_num > max_inlier_num:
            max_inlier_num = cur_inlier_num
            inlier_p1s, inlier_p2s = all_p1s[:, inlier_flags], all_p2s[:, inlier_flags]
            ransac_best_H = curH2to1
        if(i%50 == 0):
            print('RANSAC iter %d max inliers: %d' % (i, max_inlier_num))
        
    bestH = computeH(inlier_p1s, inlier_p2s)
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    #im2 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

