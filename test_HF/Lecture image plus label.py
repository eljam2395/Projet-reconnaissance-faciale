import numpy as np
import cv2

img=[]
vect_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='frontalimages_manuallyaligned_part1/'
#print(img_path+img[2]+'.jpg')

print(np.shape(img))

for i in range(0,400):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    vect_img.append(image)

fichier = open("Label.txt", "r")
Label=fichier.read()
#print(Label)
fichier.close()

print(vect_img[199])
print('lol')
print(np.vstack(vect_img[0]))

print(np.shape(vect_img))


cv2.imshow('image',vect_img[399])
cv2.waitKey(0)
cv2.destroyAllWindows()
