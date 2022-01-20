import matplotlib.image as mpimg
import numpy as np
import os
import sqlite3


def get_dummy(num):
    arr = np.zeros(9, dtype='int8')
    arr[num] = 1
    return arr


def add_image(img_path):
    """
    :param img_path:
    :return: a 1-dimension array xith the grey value of the image
    """
    img = mpimg.imread(img_path)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    img_grey_array = np.array(imgGray).reshape(imgGray.shape[0] * imgGray.shape[1])
    return img_grey_array


def load_data(directory, db_name):
    con = sqlite3.connect(db_name)
    cur = con.cursor()

    X = []
    y = []

    for file in os.listdir(directory):
        cur.execute("""SELECT type FROM eyes WHERE pic_name = ?""", (file,))
        typ = cur.fetchone()
        dummy_typ = get_dummy(typ)
        X.append(add_image(directory + "/" + file))
        y.append(dummy_typ)

    return np.array(X), np.array(y)
