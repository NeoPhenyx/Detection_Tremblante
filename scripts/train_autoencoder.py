import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
import glob
import imghdr

# === CONFIGURATION GLOBALE ===
BASE_DIR = r"D:\Projets\TP-FORMATIVE\Detection_Tremblante"
SAIN_DIR = os.path.join(BASE_DIR, "images", "Sain")
MALADE_DIR = os.path.join(BASE_DIR, "images", "Malade", "all")
REALISME_DIR = os.path.join(BASE_DIR, "images_realisme")
MODEL_DIR = os.path.join(BASE_DIR, "Detection_Tremblante", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

AE_PATH = os.path.join(MODEL_DIR, "autoencoder_tremblante.pkl")
REALISME_PATH = os.path.join(MODEL_DIR, "realisme_classifier.h5")

# === Vérif des dossiers ===
for d in [SAIN_DIR, MALADE_DIR, REALISME_DIR]:
    if not os.path.exists(d):
        raise FileNotFoundError(f"❌ Dossier introuvable : {d}")

# === Fonction nettoyage ===
def clean_non_images(root_dir):
    removed = 0
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            file_type = imghdr.what(path)
            if file_type not in ("jpeg", "png", "bmp", "gif"):
                try:
                    os.remove(path)
                    removed += 1
                    print(f"🗑️ Supprimé : {path}")
                except Exception as e:
                    print(f"⚠️ Erreur suppression {path}: {e}")
    if removed == 0:
        print(f"✅ Aucun fichier non-image détecté dans {root_dir}")
    else:
        print(f"🧹 {removed} fichier(s) non-image supprimé(s) dans {root_dir}")

# === Nettoyage images réalisme ===
print("🧼 Vérification et nettoyage des images dans :", REALISME_DIR)
clean_non_images(REALISME_DIR)

# === Dataset réalisme ===
img_height, img_width = 128, 128
batch_size = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    REALISME_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=(img_height, img_width), batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    REALISME_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=(img_height, img_width), batch_size=batch_size
)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# === CNN réalisme ===
model_realisme = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_realisme.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("🚀 Entraînement du modèle de réalisme...")
model_realisme.fit(train_ds, validation_data=val_ds, epochs=20)
model_realisme.save(REALISME_PATH)
print(f"✅ Modèle de réalisme sauvegardé : {REALISME_PATH}\n")

# ===================================================================
#  ENTRAÎNEMENT DE L’AUTOENCODEUR DE TREMBLANTE
# ===================================================================
print("=== ENTRAÎNEMENT DE L’AUTOENCODEUR DE TREMBLANTE ===")

def load_images_from_folder(folder):
    data = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in glob.glob(os.path.join(folder, "**", ext), recursive=True):
            img = image.load_img(path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            data.append(img_array)
    return np.array(data, dtype=np.float32)

train_data_sain = load_images_from_folder(SAIN_DIR)
train_data_malade = load_images_from_folder(MALADE_DIR)

print(f"✅ {train_data_sain.shape[0]} images saines | {train_data_malade.shape[0]} images malades")

# === Définition autoencodeur ===
input_img = tf.keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(16, 3, activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

print("🚀 Entraînement du modèle autoencodeur...")
autoencoder.fit(train_data_sain, train_data_sain, epochs=40, batch_size=8, shuffle=True, verbose=1)
print("✅ Entraînement terminé.")

# === Calcul MSE sain/malade ===
def compute_mse(data):
    mse_list = []
    for img_arr in data:
        arr = np.expand_dims(img_arr, axis=0)
        recon = autoencoder.predict(arr, verbose=0)
        mse = np.mean(np.square(arr - recon))
        mse_list.append(mse)
    return np.array(mse_list)

mse_sain = compute_mse(train_data_sain)
mse_malade = compute_mse(train_data_malade)

print(f"\n📈 MSE sain moyen : {np.mean(mse_sain):.9f}")
print(f"📉 MSE malade moyen : {np.mean(mse_malade):.9f}")

# === Calibration automatique ===
def find_best_threshold(mse_sain, mse_malade):
    all_mse = np.concatenate([mse_sain, mse_malade])
    thresholds = np.linspace(np.min(all_mse), np.max(all_mse), 400)
    best = {"score": -1, "threshold": None, "rule": None}
    for thr in thresholds:
        # règle 1
        tprA = np.mean(mse_malade > thr)
        fprA = np.mean(mse_sain > thr)
        scoreA = tprA - fprA
        if scoreA > best["score"]:
            best = {"score": scoreA, "threshold": float(thr), "rule": "gt"}
        # règle 2
        tprB = np.mean(mse_malade < thr)
        fprB = np.mean(mse_sain < thr)
        scoreB = tprB - fprB
        if scoreB > best["score"]:
            best = {"score": scoreB, "threshold": float(thr), "rule": "lt"}
    return best

best = find_best_threshold(mse_sain, mse_malade)
threshold = best["threshold"]
rule = best["rule"]

print(f"🚪 Seuil automatique : {threshold:.9f} | Règle : malade si MSE {'>' if rule=='gt' else '<'} seuil")

# === Sauvegarde modèle ===
model_dict = {
    "architecture": autoencoder.to_json(),
    "weights": autoencoder.get_weights(),
    "input_shape": autoencoder.input_shape,
    "loss": "mse",
    "optimizer": "adam",
    "threshold": threshold,
    "rule": rule
}

with open(AE_PATH, "wb") as f:
    pickle.dump(model_dict, f)

print(f"💾 Autoencodeur sauvegardé : {AE_PATH}")
print("🎯 Les deux modèles sont prêts à être utilisés dans Flask ✅")
