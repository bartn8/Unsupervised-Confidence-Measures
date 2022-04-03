from pyUCM import confidence_measure
from pyrSGM import census5x5_SSE, median3x3_SSE, costMeasureCensus5x5_xyd_SSE, matchWTA_SSE, aggregate_SSE, subPixelRefine
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

left = cv2.imread(os.path.join(".", "view1.png"), cv2.IMREAD_GRAYSCALE)
right = cv2.imread(os.path.join(".", "view5.png"), cv2.IMREAD_GRAYSCALE)

left = rsgm_crop_left(left)
right = rsgm_crop_left(right)

h,w = left.shape[:2]
h,w = int(h), int(w)
dispCount = int(128)

startTimeSGM = time.time()

dsiLR = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiRL = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiLL = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiRR = np.zeros((h,w,dispCount), dtype=np.uint16)

leftCensus = np.zeros(left.shape, dtype=np.uint32)
rightCensus = np.zeros(left.shape, dtype=np.uint32)

census5x5_SSE(left, leftCensus, w, h)
census5x5_SSE(right, rightCensus, w, h)

costMeasureCensus5x5_xyd_SSE(leftCensus, rightCensus, dsiLR, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(rightCensus, leftCensus, dsiRL, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(leftCensus, leftCensus, dsiLL, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(rightCensus, rightCensus, dsiRR, w, h, dispCount, 4)

dsiLRAgg = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiRLAgg = np.zeros((h,w,dispCount), dtype=np.uint16)

aggregate_SSE(left, dsiLR, dsiLRAgg, w, h, dispCount, 7, 17, 0.25, 50)
aggregate_SSE(right, dsiRL, dsiRLAgg, w, h, dispCount, 7, 17, 0.25, 50)

dispImgLeft = np.zeros((h,w), dtype=np.float32)
dispImgRight = np.zeros((h,w), dtype=np.float32)

matchWTA_SSE(dsiLRAgg, dispImgLeft, w,h,dispCount,float(0.95))
subPixelRefine(dsiLRAgg, dispImgLeft, w,h,dispCount,0)

matchWTA_SSE(dsiRLAgg, dispImgRight, w,h,dispCount,float(0.95))
subPixelRefine(dsiRLAgg, dispImgRight, w,h,dispCount,0)

dispImgLeftfiltered = np.zeros((h,w), dtype=np.float32)
dispImgRightfiltered = np.zeros((h,w), dtype=np.float32)

median3x3_SSE(dispImgLeft, dispImgLeftfiltered, w, h)
median3x3_SSE(dispImgRight, dispImgRightfiltered, w, h)

bad = 3

dsiLRAgg = dsiLRAgg.astype(np.float32)
dsiRLAgg = dsiRLAgg.astype(np.float32)
dsiLL = dsiLL.astype(np.float32)
dsiRR = dsiRR.astype(np.float32)
confidences = np.zeros((3,3,3), dtype=np.float32)

choices = confidence_measure(confidences,left, right, dispImgLeftfiltered, dispImgRightfiltered, dsiLRAgg, dsiRLAgg, dsiLL, dsiRR, bad, w, h, int(0), int(127), float(0.3), "lrc uc dbl apkr med wmn", "lrc uc apkr wmn")
print(choices)