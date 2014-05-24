import slic
import cv2

cv2.imwrite('dog_slic_py.png',
	slic.oversegmentate(cv2.imread('dog.png')))
