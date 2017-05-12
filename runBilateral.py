import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def bilateral(im, sigma_s = 5, sigma_r = 0.1):
    radius = 3 * sigma_s

    result = np.zeros(im.shape, dtype = np.float32)

    Gs = np.zeros((2 * radius + 1, 2 * radius + 1), dtype = np.float32)
    for u in range(- radius, radius + 1):
        for v in range(- radius, radius + 1):
            Gs[u + radius][v + radius] = np.exp(- (u ** 2 + v ** 2) / sigma_s ** 2)
    
    img = cv2.copyMakeBorder(im, radius, radius, radius, radius, cv2.BORDER_REPLICATE)

    for i in range(radius, img.shape[0] - radius):
        for j in range(radius, img.shape[1] - radius):
            numerator = np.sum(np.exp( - (img[i - radius : i + radius + 1, j - radius : j + radius + 1] - img[i][j]) ** 2 / (sigma_r ** 2)) * Gs * img[i - radius : i + radius + 1, j - radius : j + radius + 1])
            denominator = np.sum(np.exp( - (img[i - radius : i + radius + 1, j - radius : j + radius + 1] - img[i][j]) ** 2 / (sigma_r ** 2)) * Gs)

            result[i - radius][j - radius] = numerator / denominator

    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    
    sigma_s = 5
    sigma_r = 0.5
    result = bilateral(im, sigma_s, sigma_r)

    cv2.imshow('output', result)
    cv2.waitKey(0)