import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def jpegCompress(image, quantmatrix):
    H = np.size(image, 0)
    W = np.size(image, 1)

    H8 = H / 8 if H % 8 == 0 else H / 8 + 1
    W8 = W / 8 if W % 8 == 0 else W / 8 + 1

    img = cv2.copyMakeBorder(image, 0, H8 * 8 - H, 0, W8 * 8 - W, cv2.BORDER_CONSTANT)


    blocks = []
    for i in range(H8):
        for j in range(W8):
            blocks.append(img[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] - 128)

    dcts = []
    for block in blocks:
        dcts.append(np.round(cv2.dct(block)))

    quantizations = []
    for dct in dcts:
        quantizations.append(np.round(dct / quantmatrix))

    imgs = []
    for quantization in quantizations:
        imgs.append(cv2.idct(quantization * quantmatrix) + 128)

    result = np.zeros(img.shape, dtype = np.float32)

    for i in range(H8):
        for j in range(W8):
            result[i * 8 : i * 8 + 8, j * 8 : j * 8 + 8] = imgs[i * W8 + j]

    result = result[ : H, : W]
    
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp', cv2.IMREAD_GRAYSCALE)
    im = np.float32(im)

    quantmatrix = sio.loadmat('./misc/quantmatrix.mat')['quantmatrix']

    out = jpegCompress(im, quantmatrix)

    cv2.imshow('output', out / 255.0)
    cv2.waitKey(0)
