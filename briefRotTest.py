import numpy as np
import cv2
from BRIEF import briefLite, briefMatch, plotMatches

def get_rotated_image(im, degree):
    H, W, C = im.shape
    stage_size = int((H**2+W**2)**0.5+1)

    anchor_src = np.array([[W/2.0, H/2.0], [W/2.0, 0], [0, H/2.0]], dtype=np.float32)

    d_theta = np.deg2rad(0)

    dst_center = [stage_size/2.0, stage_size/2.0]

    r1, r2 = H/2.0, W/2.0
    theta1, theta2 = -(np.pi/2.0) + d_theta, -(np.pi) + d_theta
    dst_a1 = [r1*np.cos(theta1)+dst_center[0], r1*np.sin(theta1)+dst_center[1]]
    dst_a2 = [r2*np.cos(theta2)+dst_center[0], r2*np.sin(theta2)+dst_center[1]]

    anchor_dst = np.array([dst_center, dst_a1, dst_a2], dtype=np.float32)

    M = cv2.getAffineTransform(anchor_src, anchor_dst)
    im_rot = cv2.warpAffine(im, M, (stage_size, stage_size))
    return im_rot, M


im = cv2.imread('../data/model_chickenbroth.jpg')
im_rot, M = get_rotated_image(im, 0)

locs1, desc1 = briefLite(im)
locs2, desc2 = briefLite(im_rot)
matches = briefMatch(desc1, desc2)

im_rot_matched_locs = locs2[matches[:, 1], :]
# homogeneous coord
im_rot_matched_locs[:, 2] = 1
# (2*N)
im_rot_reprojected_locs = cv2.invertAffineTransform(M) @ np.transpose(im_rot_matched_locs, (1, 0))
# (N*2)
im_rot_reprojected_locs = im_rot_reprojected_locs.transpose((1, 0))
# (N*2)
im_matched_locs = locs1[matches[:, 0], :][:, 0:2]

dists = np.sum((im_matched_locs - im_rot_reprojected_locs)**2, axis=1)
n_correct = np.sum(dists < 25)
n_matches = matches.shape[0]

accuracy = n_correct * 1.0 / n_matches
print(accuracy)
plotMatches(im,im_rot,matches,locs1,locs2)