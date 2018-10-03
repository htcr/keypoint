import numpy as np
import cv2

im = cv2.imread('../data/model_chickenbroth.jpg')
H, W, C = im.shape
stage_size = int((H**2+W**2)**0.5+1)

# im_rot = np.zeros((stage_size, stage_size, C), dtype=np.uint8)

anchor_src = np.array([[W/2.0, H/2.0], [W/2.0, 0], [0, H/2.0]], dtype=np.float32)

d_theta = np.deg2rad(60)

dst_center = [stage_size/2.0, stage_size/2.0]

r1, r2 = H/2.0, W/2.0
theta1, theta2 = -(np.pi/2.0) + d_theta, -(np.pi) + d_theta
dst_a1 = [r1*np.cos(theta1)+dst_center[0], r1*np.sin(theta1)+dst_center[1]]
dst_a2 = [r2*np.cos(theta2)+dst_center[0], r2*np.sin(theta2)+dst_center[1]]

anchor_dst = np.array([dst_center, dst_a1, dst_a2], dtype=np.float32)

M = cv2.getAffineTransform(anchor_src, anchor_dst)
im_rot = cv2.warpAffine(im, M, (stage_size, stage_size))

cv2.imwrite('rot_test.png', im_rot)