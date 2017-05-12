import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def harrisdetector(image, k, t):
    ptr_x = []
    ptr_y = []

    kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Ix = cv2.filter2D(image, -1, kernel)
    kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    Iy = cv2.filter2D(image, -1, kernel)

    XX = Ix * Ix
    XY = Ix * Iy
    YY = Iy * Iy

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x1 = i - k if i - k >= 0 else 0
            x2 = i + k + 1 if i + k + 1 < image.shape[0] else image.shape[0] - 1
            y1 = j - k if j - k >= 0 else 0
            y2 = j + k + 1 if j + k + 1 < image.shape[1] else image.shape[1] - 1
            A = np.sum(XX[x1 : x2, y1 : y2])
            B = np.sum(XY[x1 : x2, y1 : y2])
            C = np.sum(YY[x1 : x2, y1 : y2])

            M = np.array([[A, B], [B, C]])
            eVal, eVec = cv2.eigen(M, True)[1 : ]
            
            if (eVal > t).all():
                ptr_x.append(j)
                ptr_y.append(i)

    result = [ptr_x, ptr_y]
    return result

if __name__ == '__main__':
    k = 8
    t = 2e5

    I = cv2.imread('img1.jpg').astype(np.float32)

    scale_factor = 0.5
    I = cv2.resize(I, (0, 0), fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_LINEAR)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)


    fr = harrisdetector(I, k, t)
    print len(fr[0])
    
    plt.imshow(I, cmap = 'gray')
    plt.scatter(x = fr[0], y = fr[1], c = 'r', s = 40) 
    plt.show()  