import os 
import dlib
import cv2
import numpy as np
import random as rd
import time as tm
from math import floor,ceil

# Point posé par dlib
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

NOZE_POINTS=[28,29]

def position(c):
    switch={'HG':(0,0),
    'HM':(1,0),
    'HD':(2,0),
    'MG':(0,1),
    'MM':(1,1),
    'MD':(2,1),
    'BG':(0,2),
    'BM':(1,2),
    'BD':(2,2)}
    return switch.get(c,'You have to choose between HG,HM,HD,MG,MM,MD,BG,BM,BD')

webcam = cv2.VideoCapture(0)   
file_name = os.path.abspath('')
cwd = os.path.abspath(file_name)
model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)
face_detector = dlib.get_frontal_face_detector()
margin = 10
i=0

while (True) :
    side=input("Choisissez entre : HG,HM,HD,MG,MM,MD,BG,BM,BD\n")
    a=tm.time()
    i=0
    compteur = rd.randrange(0,400000000000000000000000000000000000)

    while i<200 :
        try :
            _, frame_o = webcam.read()
            frame=frame_o.copy()
            faces = face_detector(frame)
            landmarks = predictor(frame, faces[0])


            region_r = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])
            region_l = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
            
            region_l = region_l.astype(np.int32)
            region_r = region_r.astype(np.int32)

            min_x_r = np.min(region_r[:, 0]) - margin
            max_x_r = np.max(region_r[:, 0]) + margin
            min_y_r = np.min(region_r[:, 1]) - margin
            max_y_r = np.max(region_r[:, 1]) + margin
            
            min_x_l = np.min(region_l[:, 0]) - margin
            max_x_l = np.max(region_l[:, 0]) + margin
            min_y_l = np.min(region_l[:, 1]) - margin
            max_y_l = np.max(region_l[:, 1]) + margin


            min_y=min(min_y_l,min_y_r)
            max_y=max(max_y_l,max_y_r)
            min_x=min(min_x_l,min_x_r)
            max_x=max(max_x_l,max_x_r)

            width=max_x-min_x

            height=max_y-min_y


            #making sure all pictures are in one shapes
            if width>300 :

                delta=width-300
                max_x-=floor(delta/2)
                min_x+=ceil(delta/2)
            
            if width<300 :

                delta=300-width
                max_x+=floor(delta/2)
                min_x-=ceil(delta/2)
             
            if height<80 :

                delta=80-height
                max_y+=floor(delta/2)
                min_y-=ceil(delta/2)
            
            if height>80 :

                delta=height-80
                max_y-=floor(delta/2)
                min_y+=ceil(delta/2)
            
            
            picture = frame[min_y:max_y,min_x:max_x]  
            filename = "data/"+side+"/image"+str(compteur+i)+'.png'
            
            cv2.imwrite(filename,picture)

            i+= 1

            print('photo prise ! <3'+str(i))
            b=tm.time()
            c=b-a
            print("Le temps de prise d'une photo est de :{}".format(c))
        except IndexError :
            ()
        if cv2.waitKey(1) == 27:
            break