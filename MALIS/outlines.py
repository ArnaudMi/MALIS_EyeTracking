import os
import dlib
import numpy as np
import time
from math import floor, ceil

# Point posé par dlib
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]


file_name = os.path.abspath('')
cwd = os.path.abspath(file_name)
model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)
face_detector = dlib.get_frontal_face_detector()
margin = 10
i = 0

main_dir = "Data/raw/"
target_dir = "Data/outlines/"

for r, d, f in os.walk(main_dir):
    for file_name in f:
        start_time = time.time()

        with open(r + "/" + file_name, "r") as file:
            frame = file
            side = r.split('/')[-1]

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

            min_y = min(min_y_l, min_y_r)
            max_y = max(max_y_l, max_y_r)
            min_x = min(min_x_l, min_x_r)
            max_x = max(max_x_l, max_x_r)

            width = max_x - min_x

            height = max_y - min_y

            # making sure all pictures are in one shapes
            if width > 300:
                delta = width - 300
                max_x -= floor(delta / 2)
                min_x += ceil(delta / 2)

            if width < 300:
                delta = 300 - width
                max_x += floor(delta / 2)
                min_x -= ceil(delta / 2)

            if height < 80:
                delta = 80 - height
                max_y += floor(delta / 2)
                min_y -= ceil(delta / 2)

            if height > 80:
                delta = height - 80
                max_y -= floor(delta / 2)
                min_y += ceil(delta / 2)

            picture = frame[min_y:max_y, min_x:max_x]
            newfilename = target_dir + "/" + side + "/" + file_name

            cv2.imwrite(newfilename, picture)

            print("Processed in ", round(time.time() - start_time, 4), "seconds")






