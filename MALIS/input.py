from __future__ import division
import cv2
import dlib
import pyautogui
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image

LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

ajust = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def concatenate(image1, image2):
    c1 = image1.shape
    c2 = image2.shape
    a1, b1 = c1[0], c1[1]
    a2, b2 = c2[0], c2[1]
    a = max(a1, a2)
    imagefinale = np.array([[[255, 255, 255]] * (b1 + b2)] * a)
    imagefinale.reshape((a, b1 + b2, 3))
    for k in range(a1):
        for p in range(b1):
            imagefinale[k][p] = image1[k][p]
    for j in range(a2):
        for i in range(b2):
            imagefinale[j][i + b1] = image2[j][i]
    return imagefinale


webcam = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
file_name = os.path.abspath('')
cwd = os.path.abspath(file_name)
model_path = os.path.abspath(os.path.join(cwd, "gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)
margin = 10

PATH_l = './modele_1oeuil.pth'
model_l = models.resnet18(pretrained=True)
num_ftrs = model_l.fc.in_features
model_l.fc = nn.Linear(num_ftrs, 9)

model_l = model_l.to(device)
model_l.load_state_dict(torch.load(PATH_l))
model_l.eval()

PATH_r = './modele_1oeuildroit.pth'
model_r = models.resnet18(pretrained=True)
num_ftrs = model_r.fc.in_features
model_r.fc = nn.Linear(num_ftrs, 9)

model_r = model_r.to(device)
model_r.load_state_dict(torch.load(PATH_r))
model_r.eval()

class_names = ['bd', 'bg', 'bd', 'hd', 'hg', 'hm', 'md', 'mg', 'mm']

while True:
    try:
        cpt_temps = 0
        results_l = torch.zeros([1, 9])
        results_r = torch.zeros([1, 9])
        while cpt_temps < 3:
            _, frame_o = webcam.read()
            frame = frame_o.copy()
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
            frame_r = frame[min_y_r:max_y_r, min_x_r:max_x_r]

            min_x_l = np.min(region_l[:, 0]) - margin
            max_x_l = np.max(region_l[:, 0]) + margin
            min_y_l = np.min(region_l[:, 1]) - margin
            max_y_l = np.max(region_l[:, 1]) + margin
            frame_l = frame[min_y_l:max_y_l, min_x_l:max_x_l]
            frame_c = concatenate(frame_l, frame_r)

            """
            np.resize(frame_l, 256)
            image_l = torch.tensor(frame_l)
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_l)
            
            np.resize(frame_r, 256)
            image_r = torch.tensor(frame_r)
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_r)
            """
            image_r = Image.fromarray(frame_r)
            image_l = Image.fromarray(frame_l)
            image_r = ajust(image_r)
            image_l = ajust(image_l)
            image_r.unsqueeze_(0)
            image_l.unsqueeze_(0)
            image_r = image_r.to(device)
            image_l = image_l.to(device)

            # image_c = ajust(frame_c)

            outputs_l = model_l(image_l)  # on a un tenseur normalement
            results_l = results_l + outputs_l

            outputs_r = model_r(image_r)  # on a un tenseur normalement
            results_r = results_r + outputs_r

            cpt_temps += 1

        _, preds = torch.max(results_r + results_l, 1)
        verdict = class_names[preds]

        if verdict == 'bg':
            pyautogui.keyDown('s')
            time.sleep(0.75)
            pyautogui.keyUp('s')
            pyautogui.keyDown('q')
            time.sleep(0.75)
            pyautogui.keyUp('q')
            # ♣pyautogui.alert('bg') # Make an alert box appear and pause the program until OK is clicked.
        if verdict == 'bd':
            pyautogui.keyDown('s')
            time.sleep(0.75)
            pyautogui.keyUp('s')
            pyautogui.keyDown('d')
            time.sleep(0.75)
            pyautogui.keyUp('d')
        if verdict == 'bm':
            pyautogui.keyDown('s')
            time.sleep(0.75)
            pyautogui.keyUp('s')
        if verdict == 'md':
            pyautogui.keyDown('d')
            time.sleep(0.75)
            pyautogui.keyUp('d')
        if verdict == 'mm':
            time.sleep(0.75)
        if verdict == 'mg':
            pyautogui.keyDown('q')
            time.sleep(0.75)
            pyautogui.keyUp('q')
        if verdict == 'hg':
            pyautogui.keyDown('z')
            time.sleep(0.75)
            pyautogui.keyUp('z')
            pyautogui.keyDown('q')
            time.sleep(0.75)
            pyautogui.keyUp('q')
        if verdict == 'hd':
            pyautogui.keyDown('z')
            time.sleep(0.75)
            pyautogui.keyUp('z')
            pyautogui.keyDown('d')
            time.sleep(0.75)
            pyautogui.keyUp('d')
        if verdict == 'hm':
            pyautogui.keyDown('z')
            time.sleep(0.75)
            pyautogui.keyUp('z')
        # faire de meme pour les autres.

    except IndexError:
        ()
    if cv2.waitKey(1) == 27:
        break
