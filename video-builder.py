import os
import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('qual_run.avi', fourcc, 10.0, (640, 480))

for filename in sorted(os.listdir('prepped_vid_images/')):
    if int(filename[:-4]) <= 1010:
        out.write(cv2.imread('prepped_vid_images/' + filename))
        print('writing ' + filename)

out.release()
