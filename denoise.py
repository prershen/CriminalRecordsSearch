from cv2 import cv2
import numpy as np
import face_recognition
from matplotlib import pyplot as plt

img = cv2.imread('//home//prershen//Downloads//blur.jpeg')
img = np.array(img, dtype=np.uint8)
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.fastNlMeansDenoisingColored(converted_img, None, 10, 10, 7, 21)
cv2.imwrite('denoise.jpg',dst)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()

known_image = face_recognition.load_image_file('/home/prershen/Downloads/CUHK_training_sketch/sketch/f2-005-01-sz1.jpg')
#unknown_image = face_recognition.load_image_file(dst)
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(dst)[0]
results = face_recognition.compare_faces([biden_encoding], unknown_encoding,tolerance=0.6)
print(results[0])