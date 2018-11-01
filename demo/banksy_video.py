import cv2
import time

# #-----Reading the image-----------------------------------------------------
# img = cv2.imread('./images/tangana.jpg', 1)
#
# #-----Converting image to LAB Color model-----------------------------------
# lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#
# #-----Splitting the LAB image to different channels-------------------------
# l, a, b = cv2.split(lab)
#
# #-----Applying CLAHE to L-channel-------------------------------------------
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
# cl = clahe.apply(l)
# ret,cl = cv2.threshold(cl,127,255,cv2.THRESH_BINARY)
# cv2.imshow('CLAHE output', cl)
# # print(cl.shape)
# cv2.waitKey(0)

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = cv2.imread('./ter.png', 1)
start_time = time.time()
predictions = coco_demo.run_on_opencv_image(image)
print("Time: {:.2f} s / img".format(time.time() - start_time))

# print("Time: {:.2f} s / img".format(time.time() - start_time))
cv2.imshow("COCO detections", predictions)
cv2.waitKey(0)
