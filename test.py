from pyUCM import confidence_measure
from pyrSGM import census5x5_SSE, median3x3_SSE, costMeasureCensus5x5_xyd_SSE, matchWTA_SSE, matchWTARight_SSE, aggregate_SSE, subPixelRefine
import os
import time
import cv2
import numpy as np

def rsgm_crop_left(img):
    _,w = img.shape[:2]
    if w % 16 != 0:
        c = w % 16
        img = img[:, c:]
    return img

leftc = cv2.imread(os.path.join(".", "view1.png"), cv2.IMREAD_COLOR)
rightc = cv2.imread(os.path.join(".", "view5.png"), cv2.IMREAD_COLOR)

leftc = rsgm_crop_left(leftc)
rightc = rsgm_crop_left(rightc)

left = cv2.cvtColor(leftc, cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(rightc, cv2.COLOR_BGR2GRAY)

h,w = left.shape[:2]
h,w = int(h), int(w)
dispCount = int(128)

startTimeSGM = time.time()

dsiLR = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiLL = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiRR = np.zeros((h,w,dispCount), dtype=np.uint16)

leftCensus = np.zeros(left.shape, dtype=np.uint32)
rightCensus = np.zeros(left.shape, dtype=np.uint32)

census5x5_SSE(left, leftCensus, w, h)
census5x5_SSE(right, rightCensus, w, h)

costMeasureCensus5x5_xyd_SSE(leftCensus, rightCensus, dsiLR, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(leftCensus, leftCensus, dsiLL, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(rightCensus, rightCensus, dsiRR, w, h, dispCount, 4)

dsiLRAgg = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiLLAgg = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiRRAgg = np.zeros((h,w,dispCount), dtype=np.uint16)

aggregate_SSE(left, dsiLR, dsiLRAgg, w, h, dispCount, 7, 17, 0.25, 50)
aggregate_SSE(left, dsiLL, dsiLLAgg, w, h, dispCount, 7, 17, 0.25, 50)
aggregate_SSE(right, dsiRR, dsiRRAgg, w, h, dispCount, 7, 17, 0.25, 50)

dispImgLeft = np.zeros((h,w), dtype=np.float32)
dispImgRight = np.zeros((h,w), dtype=np.float32)

matchWTA_SSE(dsiLRAgg, dispImgLeft, w,h,dispCount,float(0.95))
subPixelRefine(dsiLRAgg, dispImgLeft, w,h,dispCount,0)

matchWTARight_SSE(dsiLRAgg, dispImgRight, w,h,dispCount,float(0.95))

dispImgLeftfiltered = np.zeros((h,w), dtype=np.float32)
dispImgRightfiltered = np.zeros((h,w), dtype=np.float32)

median3x3_SSE(dispImgLeft, dispImgLeftfiltered, w, h)
median3x3_SSE(dispImgRight, dispImgRightfiltered, w, h)

bad = 5

dsiLRAgg = dsiLRAgg.astype(np.float32)
dsiLL = dsiLL.astype(np.float32)
dsiRR = dsiRR.astype(np.float32)

pconf = np.zeros((h,w), dtype=np.float32)
nconf = np.zeros((h,w), dtype=np.float32)

confidence_measure(pconf, nconf, left, right, dispImgLeftfiltered, dispImgRightfiltered, dsiLRAgg, dsiLLAgg, dsiRRAgg, bad, w, h, int(0), int(127), float(0.6), "lrc uc dbl apkr med wmn", "lrc uc apkr wmn")


cv2.imwrite("myoutput/disp1.png", dispImgLeftfiltered.astype(np.uint8))
cv2.imwrite("myoutput/disp5.png", dispImgRightfiltered.astype(np.uint8))
cv2.imwrite("myoutput/positive_samples.png", pconf.astype(np.uint8))
cv2.imwrite("myoutput/negative_samples.png", nconf.astype(np.uint8))

leftc[pconf >0, 1] = 255
leftc[nconf >0, 2] = 255

cv2.imwrite("myoutput/rgb_samples.png", leftc.astype(np.uint8))