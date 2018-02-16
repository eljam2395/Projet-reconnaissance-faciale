import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=720, height=1080)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)
def show_frame():

    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = gray
    lmain.configure(image=gray)
    lmain.after(10, show_frame)
    #b = tk.Button(frame, text="OK", command=cap.release())
    #b.grid(row=610, column=1,padx=11,pady=3)



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


'''
slogan = tk.Button(lmain,
                   text="Hello",
                   command=cap.release())
#slogan.pack(side=tk.LEFT)
'''
show_frame()  #Display 2



window.mainloop()  #Starts GUI


