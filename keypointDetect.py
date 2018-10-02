import numpy as np
import cv2

eps = 0.00001

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = gaussian_pyramid[:, :, 1:] - gaussian_pyramid[:, :, 0:-1]
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    H, W, C = DoG_pyramid.shape
    gradients = np.zeros((H, W, 4*C), dtype=np.float32)
    for c in range(C):
        gx = cv2.Sobel(DoG_pyramid[:, :, c], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(DoG_pyramid[:, :, c], cv2.CV_32F, 0, 1)
        # Dxx, Dxy, Dyx, Dyy
        gradients[:, :, 4*c  ] = cv2.Sobel(gx, cv2.CV_32F, 1, 0)
        gradients[:, :, 4*c+1] = cv2.Sobel(gx, cv2.CV_32F, 0, 1)
        gradients[:, :, 4*c+2] = cv2.Sobel(gy, cv2.CV_32F, 1, 0)
        gradients[:, :, 4*c+3] = cv2.Sobel(gy, cv2.CV_32F, 0, 1)
        
    principal_curvature = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        principal_curvature[:, :, c] = \
            (gradients[:, :, 4*c] + gradients[:, :, 4*c+3])**2 / \
            (gradients[:, :, 4*c]*gradients[:, :, 4*c+3] - gradients[:, :, 4*c+1]*gradients[:, :, 4*c+2] + eps)
    
    return np.clip(principal_curvature, -100, 100)

def get_index(C, Ih, Iw, Kh, Kw, Sh, Sw, Ph, Pw):
    # get C, H, W indices required to sample
    # an array of (C, Ih, Iw) into (C, Oh*Ow, Kh*Kw)
    Oh = int((Ih + 2*Ph - Kh) / Sh + 1)
    Ow = int((Iw + 2*Pw - Kw) / Sw + 1)
    # Sample along H
    ih = np.repeat(np.arange(Kh), Kw)
    ih = np.tile(ih, Ow)
    ih_step = np.arange(Oh) * Sh
    ih = ih.reshape(1, -1) + ih_step.reshape(-1, 1)
    ih = ih.reshape(-1)
    ih = np.tile(ih, C)
    # Sample along W
    iw = np.tile(np.arange(Kw), Kh)
    iw_step = np.arange(Ow) * Sw
    iw = iw.reshape(1, -1) + iw_step.reshape(-1, 1)
    iw = iw.reshape(-1)
    iw = np.tile(iw, Oh*C)
    # Sample along C
    ic = np.repeat(np.arange(C), Oh*Ow*Kh*Kw)
    return ic, ih, iw


def get_local_extrema_flag(DoGPyramid):
    # input: (H, W, C)
    # output: (H, W, C), bool, Ture if is extrema
    
    # first, decide if spacial extrema
    # rearrange to get argmax/argmin, if argmax/min == 4 then it is
    # because indices for 3x3 neighborhood are: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    # HWC->CHW
    tensor = np.transpose(DoGPyramid, (2, 0, 1))
    C, H, W = tensor.shape
    spatial_padded = np.pad(tensor, ((0, 0), (1, 1), (1, 1)), mode='constant')
    ic, ih, iw = get_index(C, H, W, 3, 3, 1, 1, 1, 1)
    # CHW->C*(Oh*Ow)*(Kh*Kw)
    spatial_sampled = spatial_padded[ic, ih, iw].reshape(C, H*W, 9)
    spacial_argmaxs = np.argmax(spatial_sampled, axis=2).reshape(C, H, W)
    spacial_argmins = np.argmin(spatial_sampled, axis=2).reshape(C, H, W)

    # next, decide if scale-wise extrema
    # we look the tensor from the 'side', this is equivalent to a 
    # 1x3 max pooling with stride=1 and w_pad = 1, where the original
    # channel and width are exchanged.
    # CHW->WHC
    tensor_t = np.transpose(tensor, (2, 1, 0))
    scale_padded = np.pad(tensor_t, ((0, 0), (0, 0), (1, 1)), mode='constant')
    iw, ih, ic = get_index(W, H, C, 1, 3, 1, 1, 0, 1)
    scale_sampled = scale_padded[iw, ih, ic].reshape(W, H*C, 3)
    scale_argmaxs = np.argmax(scale_sampled, axis=2).reshape(W, H, C).transpose((2, 1, 0))
    scale_argmins = np.argmin(scale_sampled, axis=2).reshape(W, H, C).transpose((2, 1, 0))

    is_extrema = ((spacial_argmaxs==4)*(scale_argmaxs==1)) + \
        ((spacial_argmins==4)*(scale_argmins==1))

    return is_extrema.transpose((1, 2, 0))

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    is_extrama = get_local_extrema_flag(DoG_pyramid)
    th_r_satisfy = np.abs(principal_curvature) <= th_r
    th_contrast_satisfy = np.abs(DoG_pyramid) > th_contrast

    selected = is_extrama * th_r_satisfy * th_contrast_satisfy
    iy, ix, ic = np.where(selected)
    locsDoG = np.stack((ix, iy, ic), axis=1)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)
    return locsDoG, gauss_pyramid


def draw_keypoints(im, locsDoG):
    # im should be 3-channel 0-255 image
    for loc in locsDoG:
        x, y, c = loc
        cv2.circle(im, (x, y), 1, (0, 255, 0), 1)
    cv2.imwrite('keypoints.jpg', im)


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    draw_keypoints(im, locsDoG)


