import os

def count_images(base_path):
    total = 0
    for root, dirs, files in os.walk(base_path):
        total += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Nombre total d'images : {total}")
