import numpy as np
import cv2


img=[]
tab_img=[]
'''
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        '''
img_path='bdd_align_brasil/'
#print(img_path+img[2]+'.jpg')

for i in range(1,51):
    for p in range(1,15):
        
        
        if p < 10:
            nom=img_path+str(i)+"-"+"0"+str(p)+".jpg"
        else:
            nom=img_path+str(i)+"-"+str(p)+".jpg"
        image=cv2.imread(nom,0)
        tab_img.append(image)
        #print(nom)

tab_img[1]
vect_img=[]
print(np.shape(tab_img[1]))

vect_img=np.reshape(tab_img,[50*14,640*480])

np.save('Img_flatten_RF',vect_img)

#cv2.imshow('image',vect_img[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(vect_img)

#On met les labels pour définir si la personne est un homme ou une femme selon la forme suivante
# [1,0] pour un homme, [0,1] pour une femme
Label=[]


for i in range(1,51): #20
    for p in range(1,15):
        Label.append(i)

print(Label)

print(np.shape(Label))
print(type(Label[0]))


np.save('Label_RF',Label)



