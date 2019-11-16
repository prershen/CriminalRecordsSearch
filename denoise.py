from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('//home//prershen//Downloads//blur.jpeg')
img = np.array(img, dtype=np.uint8)
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.fastNlMeansDenoisingColored(converted_img, None, 10, 10, 7, 21)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()