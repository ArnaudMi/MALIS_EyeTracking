import sqlite3
import os
import shutil


def create_eye_db():
    con = sqlite3.connect("Beyes.db")
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS eyes(
            id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
            pic_name TEXT UNIQUE NOT NULL,
            type INTEGER NOT NULL
        )"""
                )

    con.commit()
    con.close()


def add_pics():
    dico_types = {"HG": 0, "HM": 1, "HD": 2, "MG": 3, "MM": 4, "MD": 5, "BG": 6, "BM": 7, "BD": 8}
    raw_dir_name = "Data/raw"
    final_dir_name = "Data/img"

    con = sqlite3.connect("Beyes.db")
    cur = con.cursor()

    for r, d, f in os.walk(raw_dir_name):
        for file in f:
            if file.endswith(".png"):
                typ = dico_types[r.split("\\")[1]]
                shutil.move(r + "/" + file, final_dir_name + "/" + file)
                cur.execute("""INSERT INTO eyes(pic_name, type) VALUES(?, ?)""", (file, typ))
    con.commit()
    con.close()
