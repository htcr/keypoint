import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    '''
    # we are sampling from a coarse polar grid, 
    # where we sample 8x more on angle than on radius
    # we sample more than nbits.
    # do numerical tricks to gain better distribution
    tn = np.sqrt(8.0*nbits)
    tr = tn - np.floor(tn)
    rn = np.sqrt(nbits/8.0)
    rr = rn - np.floor(rn)

    theta_sample_n = int(tn) + int(np.ceil((tn*rr+rn*tr+tr*rr)/rn)) + 1
    r_sample_n = int(rn)

    # x, y of center
    center_x = int(np.clip(patch_width / 2.0, 0, patch_width-1))
    
    # sample in polar space
    max_r = patch_width / 2.0
    max_theta = 2*np.pi
    step_r = max_r / r_sample_n
    step_theta = max_theta / theta_sample_n

    # we don't want the first circle of points to collapse on origin
    rs = np.arange(0, max_r, step_r) + step_r
    thetas = np.arange(0, max_theta, step_theta)

    polar_coords = np.stack(np.meshgrid(rs, thetas), axis=2).reshape(-1, 2)
    xs = polar_coords[:, 0]*np.cos(polar_coords[:, 1]) + center_x
    ys = polar_coords[:, 0]*np.sin(polar_coords[:, 1]) + center_x

    xs = np.clip(xs, 0, patch_width-1).astype(np.int64)
    ys = np.clip(ys, 0, patch_width-1).astype(np.int64)

    compareY = ys * patch_width + xs
    compareY = np.random.choice(compareY, nbits, replace=False)
    compareX = np.zeros((nbits,), dtype=np.int64)
    compareX.fill(center_x * patch_width + center_x)
    '''
    Xxs = np.random.normal(patch_width / 2.0, patch_width / 5.0, nbits)
    Xys = np.random.normal(patch_width / 2.0, patch_width / 5.0, nbits)
    Xxs = np.int64(np.clip(np.round(Xxs), 0, patch_width-1))
    Xys = np.int64(np.clip(np.round(Xys), 0, patch_width-1))

    Yxs = np.random.normal(patch_width / 2.0, patch_width / 5.0, nbits)
    Yys = np.random.normal(patch_width / 2.0, patch_width / 5.0, nbits)
    Yxs = np.int64(np.clip(np.round(Yxs), 0, patch_width-1))
    Yys = np.int64(np.clip(np.round(Yys), 0, patch_width-1))
    
    compareX = Xys * patch_width + Xxs
    compareY = Yys * patch_width + Yxs

    return compareX, compareY

brief_nbits = 256
brief_patch_width = 9
# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern(brief_patch_width, brief_nbits)
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY, patch_width=brief_patch_width):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    locs = locsDoG
    m = locs.shape[0]
    # get patch position offset w.r.t DoG locs
    p_x, p_y = np.meshgrid(np.arange(patch_width), np.arange(patch_width))
    p_x, p_y = p_x.reshape(-1), p_y.reshape(-1)
    
    #X_x, X_y = compareX % patch_width, compareX / patch_width
    #Y_x, Y_y = compareY % patch_width, compareX / patch_width
    patch_center_x, patch_center_y = patch_width // 2, patch_width // 2
    offset_p_x, offset_p_y = \
        p_x - patch_center_x, p_y - patch_center_y
    # get patch sample indices (m*patch_width**2) on gaussian pyramid
    indices = np.repeat(locs, patch_width**2, axis=0)
    p_iw, p_ih, p_ic = \
        indices[:, 0] + np.tile(offset_p_x, m), \
        indices[:, 1] + np.tile(offset_p_y, m), \
        indices[:, 2]
    # sample patchs
    # m*patch_width**2
    pad = patch_center_x
    
    # gaussian_pyramid_padded = np.pad(gaussian_pyramid, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    # p_val = gaussian_pyramid_padded[p_ih+pad, p_iw+pad, p_ic].reshape(m , -1)
    # im_padded = np.pad(cv2.GaussianBlur(im, (3, 3), 0), ((pad, pad), (pad, pad)), mode='constant')
    im_padded = np.pad(im, ((pad, pad), (pad, pad)), mode='constant')
    p_val = im_padded[p_ih+pad, p_iw+pad].reshape(m, -1)

    # now that we got pixels inside patches, we sample these pixels 
    # with compareX and compareY
    # m*patch_width**2 --> m*nbits
    X_val = p_val[:, compareX]
    Y_val = p_val[:, compareY]
    
    desc = X_val < Y_val

    return locs, desc


def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    locs, gaussian_pyramid = DoGdetector(im)
    # we use sample pattern saved as global
    locs, desc = computeBrief(im, gaussian_pyramid, locs, None, None, compareX, compareY)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        #plt.plot(x,y,'g.')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        #plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
    

def draw_pattern(patch_width, compareX, compareY):
    p = np.zeros((patch_width, patch_width, 3), dtype=np.uint8)
    for i in range(compareX.shape[0]):
        x = compareX[i]
        y = compareY[i]
        xx, xy = x % patch_width, x // patch_width
        yx, yy = y % patch_width, y // patch_width
        cv2.line(p, (xx, xy), (yx, yy), (255, 255, 255))
        p[yy, yx, :] = 255
    cv2.imwrite('test_pattern.png', p)


if __name__ == '__main__':
    # test makeTestPattern
    # compareX, compareY = makeTestPattern()
    # psize, nbits = 100, 256
    # compareX, compareY = makeTestPattern(psize, nbits)
    # draw_pattern(psize, compareX, compareY)
    
    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)  
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    # im2 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
