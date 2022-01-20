import numpy as np
import matplotlib.pyplot as plt
from LoadData import load_data
import os

directory = "Data/img"
db_name = "Beye.db"
training_length = 200

X, y = load_data(directory, db_name)


def remove_all(_directory):
    for r, d, f in os.walk(_directory):
        for file in f:
            if file.endswith(".png"):
                os.remove(r + "/" + file)

remove_all("Data/raw")