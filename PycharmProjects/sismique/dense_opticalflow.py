import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


# Loads the data
data1 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL94.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL01.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL04.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL06.txt', delimiter=',')])
data2 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x94_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x01_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x04_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x06_2.txt', delimiter=',')])


# Creates video from images (currently not working and generating an empty file)
# image_folder = 'data/imgs'
# file_name = 'seismic_evolve2.mp4'
# layers, height, width = data1.shape
# frame_size = (width, height)
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# images.sort()
#
# out = cv.VideoWriter(file_name, cv.VideoWriter_fourcc(*'mp4v'), 0.5, frame_size)
#
# img_array = []
# for filename in images:
#     img = cv.imread(os.path.join(image_folder, filename))
#     img_array.append(img)
#     out.write(img)
#
# out.release()


# Computes and plots dense optical flow (Farneback's method)
cap = cv.VideoCapture("data/img1_du637ZMP.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next


cv.destroyAllWindows()
