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

leftc = cv2.imread(os.path.join(".", "im0.png"), cv2.IMREAD_COLOR)
rightc = cv2.imread(os.path.join(".", "im1.png"), cv2.IMREAD_COLOR)

leftc = rsgm_crop_left(leftc)
rightc = rsgm_crop_left(rightc)

left = cv2.cvtColor(leftc, cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(rightc, cv2.COLOR_BGR2GRAY)

h,w = left.shape[:2]
h,w = int(h), int(w)
dispCount = int(128)

startTimeSGM = time.time()

dsiLR = np.zeros((h,w,dispCount), dtype=np.uint16)
dsiLL = np.zeros((h,w,dispCount//2), dtype=np.uint16)
dsiRR = np.zeros((h,w,dispCount//2), dtype=np.uint16)

leftCensus = np.zeros(left.shape, dtype=np.uint32)
rightCensus = np.zeros(left.shape, dtype=np.uint32)

census5x5_SSE(left, leftCensus, w, h)
census5x5_SSE(right, rightCensus, w, h)

costMeasureCensus5x5_xyd_SSE(leftCensus, rightCensus, dsiLR, w, h, dispCount, 4)
costMeasureCensus5x5_xyd_SSE(leftCensus, leftCensus, dsiLL, w, h, dispCount//2, 4)
costMeasureCensus5x5_xyd_SSE(rightCensus, rightCensus, dsiRR, w, h, dispCount//2, 4)

dsiLRAgg = np.zeros((h,w,dispCount), dtype=np.uint16)
#dsiLLAgg = np.zeros((h,w,dispCount), dtype=np.uint16)
#dsiRRAgg = np.zeros((h,w,dispCount), dtype=np.uint16)

#dsiLRAgg = dsiLR
aggregate_SSE(left, dsiLR, dsiLRAgg, w, h, dispCount, 7, 17, 0.25, 50)
#aggregate_SSE(left, dsiLL, dsiLLAgg, w, h, dispCount, 7, 17, 0.25, 50)
#aggregate_SSE(right, dsiRR, dsiRRAgg, w, h, dispCount, 7, 17, 0.25, 50)

dispImgLeft = np.zeros((h,w), dtype=np.float32)
dispImgRight = np.zeros((h,w), dtype=np.float32)

matchWTA_SSE(dsiLRAgg, dispImgLeft, w,h,dispCount,float(0.95))
subPixelRefine(dsiLRAgg, dispImgLeft, w,h,dispCount,0)

matchWTARight_SSE(dsiLRAgg, dispImgRight, w,h,dispCount,float(0.95))

dispImgLeftfiltered = np.zeros((h,w), dtype=np.float32)
dispImgRightfiltered = np.zeros((h,w), dtype=np.float32)

median3x3_SSE(dispImgLeft, dispImgLeftfiltered, w, h)
median3x3_SSE(dispImgRight, dispImgRightfiltered, w, h)

dispImgLeftfiltered = np.clip(dispImgLeftfiltered,0,255)
dispImgRightfiltered = np.clip(dispImgRightfiltered,0,255)

stopTimeSGM = time.time()

bad = 3
threshold = float(0.3)

dsiLR = np.moveaxis(dsiLR, -1 , 0)
dsiLRAgg = np.moveaxis(dsiLRAgg, -1, 0)
dsiLL = np.moveaxis(dsiLL, -1, 0)
dsiRR = np.moveaxis(dsiRR, -1, 0)

pconf = np.zeros((h,w), dtype=np.float32)
nconf = np.zeros((h,w), dtype=np.float32)

startTimeUCM = time.time()

confidence_measure(pconf, nconf, left, right, dispImgLeftfiltered, dispImgRightfiltered,
 dsiLRAgg.astype(np.float32), dsiLL.astype(np.float32), dsiRR.astype(np.float32),
  bad, w, h, int(0), int(dispCount), threshold, "lrc uc dbl apkr med wmn", "lrc uc apkr wmn", True)

stopTimeUCM = time.time()

print(f"SGM time: {(stopTimeSGM-startTimeSGM)*1000} ms, UCM time: {(stopTimeUCM-startTimeUCM)*1000} ms")

print(f"pconf range: {np.unique(pconf)}")
print(f"nconf range: {np.unique(nconf)}")

cv2.imwrite("myoutput/disp1.png", dispImgLeftfiltered.astype(np.uint8))
cv2.imwrite("myoutput/disp5.png", dispImgRightfiltered.astype(np.uint8))
cv2.imwrite("myoutput/positive_samples.png", pconf.astype(np.uint8))
cv2.imwrite("myoutput/negative_samples.png", nconf.astype(np.uint8))

leftc[pconf >0, 0] = 0
leftc[pconf >0, 1] = 255
leftc[pconf >0, 2] = 0

leftc[nconf >0, 0] = 0
leftc[nconf >0, 1] = 0
leftc[nconf >0, 2] = 255

cv2.imwrite("myoutput/rgb_samples.png", leftc.astype(np.uint8))