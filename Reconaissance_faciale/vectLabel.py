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

for i in range(1,52):
    if i==51:
        for p in range(1,11):
            if p < 10:
                nom=img_path+str(i)+"-"+"0"+str(p)+".jpg"
            else:
                nom=img_path+str(i)+"-"+str(p)+".jpg"
                
            image=cv2.imread(nom,0)
            
            img_scaled=cv2.resize(image,(360,260))
            
            #img_scaled = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            
            tab_img.append(img_scaled)
            #print(np.shape(img_scaled))
            #print(nom)
    else:
        for p in range(1,15):
            
            
            if p < 10:
                nom=img_path+str(i)+"-"+"0"+str(p)+".jpg"
            else:
                nom=img_path+str(i)+"-"+str(p)+".jpg"
                
            image=cv2.imread(nom,0)
            
            img_scaled=cv2.resize(image,(360,260))
            
            #img_scaled = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            
            tab_img.append(img_scaled)
            #print(np.shape(img_scaled))
            #print(nom)

tab_img[1]
vect_img=[]
print(np.shape(image))
print(np.shape(tab_img))

vect_img=np.reshape(tab_img,[710,260*360])
print(np.shape(vect_img[1]))

np.save('Img_flatten_RF',vect_img)

#cv2.imshow('image',tab_img[1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#print(vect_img)

#On met les labels pour définir si la personne est un homme ou une femme selon la forme suivante
# [1,0] pour un homme, [0,1] pour une femme
Label=[]


for i in range(1,52): #20
    if i==51:
        for p in range(1,11):
            lab_int=np.zeros(51)
            lab_int[i-1]=1
            Label.append(lab_int)

    else:
        for p in range(1,15):
            lab_int=np.zeros(51)
            lab_int[i-1]=1
            Label.append(lab_int)

print(np.shape(Label))

print(Label[709])


print(type(Label[49]))


np.save('Label_RF',Label)



