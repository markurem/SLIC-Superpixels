import slic
import cv2

cv2.imwrite('dog_slic_py.png',
	slic.oversegmentate(cv2.imread('dog.png')))

cv2.imwrite('dog_slic_py_reg200.png',
	slic.oversegmentate(cv2.imread('dog.png'), regularity=200))

cv2.imwrite('dog_slic_py_1000.png',
	slic.oversegmentate(cv2.imread('dog.png'), num_superpixel=1000))