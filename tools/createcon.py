import cv2
import numpy as np
import os


def getfiles(path):
        dirs = []
        for item in os.listdir(path):
#               print item[-7:]

                if item[-7:] == "ori.jpg":
                        dirs.append(item)
        return dirs

path = 'anime'

files = getfiles(path)

print(files)

for file in files:
	print file[:-8]
	imgori = cv2.imread("anime/" + file[:-7]+"ori.jpg")
	imgedge = cv2.imread("anime/" + file[:-7]+"edge.jpg")


	imgori2 = cv2.resize(imgori, (256, 256))
	imgedge2 = cv2.resize(imgedge, (256, 256))

	vis = np.concatenate((imgori2,imgedge2), axis = 1)
	cv2.imwrite("animecon/" + file[:-8] + ".png", vis)
