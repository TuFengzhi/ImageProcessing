import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def rlDeconv(B, PSF):
    maxIters = 25

    pad_w = 30
    I = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)),'edge')
    B = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)),'edge') 

    for i in range(3):
        for j in range(0, maxIters):
            I[ : , : , i] = I[ : , : , i] * cv2.filter2D((B[ : , : , i] / cv2.filter2D(I[ : , : , i], -1, cv2.flip(cv2.flip(PSF, 0), 1))), -1, PSF)
    	
    I = I[pad_w : - pad_w, pad_w : - pad_w]

    return I

if __name__ == '__main__':
    gt = cv2.imread('./misc/lena_gray.bmp').astype('double') / 255.0

    PSF = sio.loadmat('./misc/psf.mat')['PSF']

    B = cv2.filter2D(gt, -1, PSF)

    I = rlDeconv(B, PSF)
    
    cv2.imshow('B',B)
    cv2.imshow('I',I)
    cv2.waitKey(0)
