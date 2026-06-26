from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

BASE_DIR = r"D:\Projets\TP-FORMATIVE\Detection_Tremblante\Detection_Tremblante"
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder_tremblante.pkl")
REALISME_PATH = os.path.join(MODEL_DIR, "realisme_classifier.h5")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# === Chargement des modèles ===
print("📦 Chargement des modèles...")
with open(AUTOENCODER_PATH, "rb") as f:
    model_dict = pickle.load(f)

autoencoder = tf.keras.models.model_from_json(model_dict["architecture"])
autoencoder.set_weights(model_dict["weights"])
autoencoder.compile(optimizer=model_dict["optimizer"], loss=model_dict["loss"])
threshold = float(model_dict.get("threshold", 0.02))
rule = model_dict.get("rule", "gt")

realisme_model = tf.keras.models.load_model(REALISME_PATH)
yolo_model = YOLO("yolov8n.pt")

# === Fonctions ===
def preprocess_image(img_path, target_size=(128, 128)):
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def contains_sheep(img_path, min_conf=0.25):
    results = yolo_model(img_path)
    for box in results[0].boxes:
        if float(box.conf[0]) >= min_conf:
            cls_id = int(box.cls[0])
            if results[0].names[cls_id].lower() == "sheep":
                return True
    return False

# === Routes Flask ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error_msg="⚠️ Aucun fichier sélectionné.")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        if not contains_sheep(file_path):
            return render_template("index.html", error_msg="⚠️ Aucun mouton détecté dans l’image.")

        img_arr = preprocess_image(file_path)
        realism_score = float(realisme_model.predict(img_arr, verbose=0)[0][0])

        if realism_score < 0.5:
            return render_template(
                "index.html",
                original_image=url_for("static", filename=f"uploads/{filename}"),
                diagnostic="🧸 Objet non vivant détecté (figurine, jouet, statue)",
                color="gray"
            )

        reconstructed = autoencoder.predict(img_arr, verbose=0)
        mse = np.mean(np.square(img_arr - reconstructed))

        if rule == "gt":
            is_sick = mse > threshold
        else:
            is_sick = mse < threshold

        if is_sick:
            diagnostic = "🔴 Mouton probablement malade (tremblante détectée)"
            color = "red"
        else:
            diagnostic = "🟢 Mouton en bonne santé (aucune tremblante détectée)"
            color = "green"

        reconstructed_img = (reconstructed[0] * 255).astype(np.uint8)
        reconstructed_path = os.path.join(app.config["UPLOAD_FOLDER"], "reconstructed_" + filename)
        tf.keras.preprocessing.image.save_img(reconstructed_path, reconstructed_img)

        return render_template(
            "index.html",
            original_image=url_for("static", filename=f"uploads/{filename}"),
            reconstructed_image=url_for("static", filename=f"uploads/reconstructed_{filename}"),
            mse_score=round(float(mse), 6),
            diagnostic=diagnostic,
            color=color
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
