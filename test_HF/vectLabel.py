import numpy as np
import cv2


img=[]
vect_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='frontalimages_manuallyaligned_part1/'
#print(img_path+img[2]+'.jpg')

for i in range(1,201):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    vect_img.append(image)

vect_img[1]

#cv2.imshow('image',vect_img[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(vect_img)

#On met les labels pour définir si la personne est un homme ou une femme selon la forme suivante
# [1,0] pour un homme, [0,1] pour une femme
Label=[] 

for i in range(1,21): #20
    Label.append([1,0])

for i in range(1,5): #24
    Label.append([0,1])

for i in range(1,5): #28
    Label.append([1,0])

for i in range(1,3): #30
    Label.append([0,1])

for i in range(1,11): #40
    Label.append([1,0])

for i in range(1,3): #42
    Label.append([0,1])

for i in range(1,7): #48
    Label.append([1,0])
    
for i in range(1,5): #52
    Label.append([0,1])

for i in range(1,3): #54
    Label.append([1,0])

for i in range(1,5): #58
    Label.append([0,1])

for i in range(1,15): #72
    Label.append([1,0])


for i in range(1,3): #74
    Label.append([0,1])

for i in range(1,3): #76
    Label.append([1,0])

for i in range(1,3): #78
    Label.append([0,1])

for i in range(1,7): #84
    Label.append([1,0])

for i in range(1,3): #86
    Label.append([0,1])

for i in range(1,15): #100
    Label.append([1,0])

for i in range(1,3): #102
    Label.append([0,1])


for i in range(1,13): #114
    Label.append([1,0])

for i in range(1,3): #116
    Label.append([0,1])

for i in range(1,3): #118
    Label.append([1,0])
    
for i in range(1,3): #120 
    Label.append([0,1])

for i in range(1,17): #136
    Label.append([1,0])

for i in range(1,3): #138
    Label.append([0,1])

for i in range(1,5): #142
    Label.append([1,0])

for i in range(1,3): #144
    Label.append([0,1])

for i in range(1,5):#148
    Label.append([1,0])

for i in range(1,3): #150
    Label.append([0,1])

for i in range(1,5): #154
    Label.append([1,0])

for i in range(1,5): #158
    Label.append([0,1])

for i in range(1,9): #166
    Label.append([1,0])
    
for i in range(1,9): #174
    Label.append([0,1])

for i in range(1,7): #180
    Label.append([1,0])

for i in range(1,3): #182
    Label.append([0,1])

for i in range(1,9): #190
    Label.append([1,0])

for i in range(1,23): #212
    Label.append([0,1])

for i in range(1,21): #232
    Label.append([1,0])

for i in range(1,3): #234
    Label.append([0,1])

for i in range(1,3): #236
    Label.append([1,0])

for i in range(1,3): #238
    Label.append([0,1])
    
for i in range(1,3): #240
    Label.append([1,0])

for i in range(1,53): #292
    Label.append([0,1])

for i in range(1,5): #296
    Label.append([1,0])

for i in range(1,7): #302
    Label.append([0,1])

for i in range(1,7): #308
    Label.append([1,0])

for i in range(1,21): #328
    Label.append([0,1])

for i in range(1,3): #330
    Label.append([1,0])

for i in range(1,11): #340
    Label.append([0,1])

for i in range(1,3): #342
    Label.append([1,0])

for i in range(1,21): #362
    Label.append([0,1])

for i in range(1,3): #364
    Label.append([1,0])

for i in range(1,5): #368
    Label.append([0,1])

for i in range(1,9): #376
    Label.append([1,0])

for i in range(1,3): #378
    Label.append([0,1])

for i in range(1,11): #388
    Label.append([1,0])

for i in range(1,13): #400
    Label.append([0,1])
    i=i+1

print(np.shape(Label))
print(type(Label[0]))

<<<<<<< HEAD
np.save('label_test', Label)
=======
np.save('Label_test',Label)
>>>>>>> Maxime

fichier = open("vectLabel.txt", "w")
for item in Label:
    fichier.write(str(item))
    fichier.write("\n")
fichier.close()



