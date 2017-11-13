import numpy as np
import cv2

img=[]
vect_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='F:/Documents/Iris 3A/Projet 3A/Git/Projet-reconnaissance-faciale/test_HF/frontalimages_manuallyaligned_part1/'
print(img_path+img[2]+'.jpg')

for i in range(1,201):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    vect_img.append(image)


cv2.imshow('image',vect_img[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(vect_img)
