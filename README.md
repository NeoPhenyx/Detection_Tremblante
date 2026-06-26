# 🐑 Détection de la tremblante chez les moutons

Ce projet entraîne un autoencodeur convolutionnel pour reconnaître les moutons atteints de la **tremblante** à partir d'images.

## 📁 Structure
- `data/` : contient les images (train/test)
- `scripts/` : contient les scripts Python
- `models/` : contient le modèle entraîné

## 🚀 Entraînement
```
cd scripts
python train_autoencoder.py
```

## 🧠 Test d’une image
```
cd scripts
python test_autoencoder.py
```
Puis dans le script :
```python
detect_tremblante("chemin/vers/image.jpg")
```

## 🔧 Démarrage local

Suivez ces commandes depuis la racine du projet (D:\Projets\TP-FORMATIVE\Detection_Tremblante).

- Git Bash / bash :
```bash
cd /d/Projets/TP-FORMATIVE/Detection_Tremblante
# activer l'environnement virtuel
source venv/Scripts/activate
# installer les dépendances
pip install -r requirements.txt
# lancer l'application Flask
python scripts/app_flask.py
```

- PowerShell :
```powershell
cd D:\Projets\TP-FORMATIVE\Detection_Tremblante
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\app_flask.py
```

- Invite de commandes (cmd.exe) :
```cmd
cd /d D:\Projets\TP-FORMATIVE\Detection_Tremblante
venv\Scripts\activate.bat
pip install -r requirements.txt
python scripts\app_flask.py
```

Ouvrez ensuite votre navigateur sur : http://127.0.0.1:5000

Remarque : l'installation de `tensorflow` et `torch` sous Windows peut nécessiter des roues spécifiques selon votre version de Python. Si vous rencontrez des erreurs, partagez-les et je vous aiderai à les résoudre.
