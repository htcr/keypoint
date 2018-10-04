import numpy as np
import cv2
from planarH import computeH
import matplotlib.pyplot as plt

def get_sphere(data_path):
    file = open(data_path, 'r')
    lines = file.readlines()
    numbers = [[float(s) for s in line.strip().split('  ')] for line in lines]
    points = np.array(numbers, dtype=np.float32)
    return points


def compute_extrinsics(K, H):
    FI = np.linalg.inv(K) @ H
    U, L, Vt = np.linalg.svd(FI[:, 0:2])
    R_c01 = U @ np.array([[1, 0],[0, 1],[0,0]], dtype=np.float32) @ Vt
    R_c2 = np.cross(R_c01[:, 0], R_c01[:, 1])[:, np.newaxis]
    R_tmp = np.concatenate((R_c01, R_c2), axis=1)
    det = np.linalg.det(R_tmp)
    R = np.concatenate((R_c01, det*R_c2), axis=1)
    l = np.sum(FI[:, 0:2]/R[:, 0:2]) / 6.0
    t = FI[:, 2:3] / l
    return R, t


def project_extrinsics(K, W, R, t):
    p =  K @ (R @ W + t)
    p[0, :] /= p[2, :]
    p[1, :] /= p[2, :]
    return p[0:2, :]


W = np.array([[0.0, 18.2, 18.2, 0.0], [0.0, 0.0, 26.0, 26.0], [0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1]], dtype=np.float32)
X = np.array([[480, 1704, 2175, 67], [810, 781, 2217, 2286]], dtype=np.float32)
K = np.array([[3043.72, 0, 1196], [0, 3043.72, 1604], [0, 0, 1]], dtype=np.float32)

pw = W[0:2, :]
pi = X[0:2, :]

Hw2i = computeH(pi, pw)

R, t = compute_extrinsics(K, Hw2i)
print(R)
print(t)

sphere = get_sphere('../data/sphere.txt')
sphere += np.array([[6.5],[20],[6.85]], dtype=np.float32)
#sphere = W[0:3, :]

sphere_projected = project_extrinsics(K, sphere, R, t)

im = cv2.imread('../data/prince_book.jpeg')[:, :, [2,1,0]]
print(im.shape)
fig = plt.figure()
plt.imshow(im)
plt.plot(sphere_projected[0, :], sphere_projected[1, :], 'yo')


plt.show()