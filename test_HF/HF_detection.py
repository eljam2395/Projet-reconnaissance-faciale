import numpy as np
import cv2

img=[]
vect_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='frontalimages_manuallyaligned_part1/'
print(img_path+img[2]+'.jpg')

for i in range(1,201):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    vect_img.append(image)

vect_img[1]

#cv2.imshow('image',vect_img[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(vect_img)

Label=[]  #1 homme 0 femme

for i in range(1,20):
    Label.append(1)

for i in range(1,4):
    Label.append(0)

for i in range(1,4):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,10):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,6):
    Label.append(1)
    
for i in range(1,4):
    Label.append(0)

for i in range(1,2):
    Label.append(1)

for i in range(1,4):
    Label.append(0)

for i in range(1,14):
    Label.append(1)


for i in range(1,2):
    Label.append(0)

for i in range(1,2):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,6):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,14):
    Label.append(1)

for i in range(1,2):
    Label.append(0)


for i in range(1,12):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,2):
    Label.append(0)

for i in range(1,16):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,4):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,4):
    Label.append(1)


for i in range(1,4):
    Label.append(0)

for i in range(1,8):
    Label.append(1)

for i in range(1,8):
    Label.append(0)

for i in range(1,6):
    Label.append(1)
    
for i in range(1,2):
    Label.append(0)

for i in range(1,8):
    Label.append(1)

for i in range(1,22): #image106b
    Label.append(0)

for i in range(1,20):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,2):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,2):
    Label.append(1)
    
for i in range(1,52): #jusqu'à l'image 146b
    Label.append(1)

for i in range(1,4):
    Label.append(1)

for i in range(1,6):
    Label.append(0)

for i in range(1,6):
    Label.append(1)

for i in range(1,20):
    Label.append(0)

for i in range(1,2):
    Label.append(1)

for i in range(1,10):
    Label.append(0)

for i in range(1,2):
    Label.append(1)

for i in range(1,20):
    Label.append(0)

for i in range(1,2):
    Label.append(1)
for i in range(1,4):
    Label.append(0)

for i in range(1,8):
    Label.append(1)

for i in range(1,2):
    Label.append(0)

for i in range(1,10):
    Label.append(1)

i=0
for i in range(1,28): #ET C'EEEEEEEEEEST TERMINEEEEEEEEEEEEEEEE
    Label.append(0)
    i=i+1

print(np.shape(Label))


fichier = open("Label.txt", "w")
for item in Label:
    fichier.write(str(item))
    fichier.write("\n")
fichier.close()



