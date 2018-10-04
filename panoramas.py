import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    H, W = im1.shape[0:2]
    tH, tW = 2*H, 2*W
    translate_M = np.array([[1, 0, 0], [0, 1, 0*H/2.0], [0, 0, 1]], dtype=np.float32)
    pano_im2 = cv2.warpPerspective(im2, translate_M @ H2to1, (tW, tH))
    pano_im1 = cv2.warpPerspective(im1, translate_M, (tW, tH))
    pano_im = np.maximum(pano_im1, pano_im2)

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    H1, W1 = im1.shape[0:2]
    im1_corners = np.array([[0, 0], [0, H1], [W1, 0], [W1, H1]], dtype=np.float32).transpose((1, 0))
    H2, W2 = im2.shape[0:2]
    im2_corners_homo = np.array([[0, 0, 1], [0, H2, 1], [W2, 0, 1], [W2, H2, 1]], dtype=np.float32).transpose((1, 0))
    im2_corners = H2to1 @ im2_corners_homo
    im2_corners[0, :] /= im2_corners[2, :]
    im2_corners[1, :] /= im2_corners[2, :]
    im2_corners = im2_corners[0:2, :]
    corners = np.concatenate((im1_corners, im2_corners), axis=1)
    lt = np.min(corners, axis=1)
    rb = np.max(corners, axis=1)
    tW, tH = rb - lt
    
    translate_M = np.array([[1, 0, -lt[0]], [0, 1, -lt[1]], [0, 0, 1]], dtype=np.float32)
    pano_im2 = cv2.warpPerspective(im2, translate_M @ H2to1, (int(tW), int(tH)))
    pano_im1 = cv2.warpPerspective(im1, translate_M, (int(tW), int(tH)))
    pano_im = np.maximum(pano_im1, pano_im2)

    return pano_im


def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im_no_clip = imageStitching_noClip(im1, im2, H2to1)
    return pano_im_no_clip

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching(im1, im2, H2to1)
    print(H2to1)
    # np.save('../results/q6_1.npy', H2to1)
    # cv2.imwrite('../results/6_1.jpg', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # pano_im_no_clip = imageStitching_noClip(im1, im2, H2to1)
    pano_im_no_clip = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im_no_clip)
    cv2.imshow('panoramas_no_clip', pano_im_no_clip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()