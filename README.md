# PneumoScan AI - Chest X-Ray Pneumonia Detection

A deep learning web app: upload a chest X-ray -> get **NORMAL / PNEUMONIA**, a confidence score,
and a **Grad-CAM heatmap** that shows which part of the lung influenced the decision.

**Stack:** PyTorch (ResNet18, transfer learning) - FastAPI backend - HTML/CSS/JS frontend - SQLite history

> For educational purposes only. NOT a medical diagnosis tool.

## Folder structure
```
pneumonia-detector/
├── backend/
│   ├── main.py          # FastAPI server (/predict, /history, /health) + serves the website
│   ├── model_utils.py   # model, transforms, loading
│   └── gradcam.py       # Grad-CAM heatmap
├── frontend/            # index.html, style.css, script.js
├── data/                # dataset goes here (data/chest_xray/train, test)
├── models/              # trained model is saved here
├── outputs/             # graphs and metrics after training
├── download_dataset.py  # downloads the dataset
├── train.py             # trains + evaluates the model
├── run.py               # starts the web app
└── requirements.txt
```

## Step-by-step (VS Code)
**1. Open the folder** - VS Code -> File -> Open Folder -> `pneumonia-detector`. Open a terminal (Ctrl + `).
Python 3.9+ is needed.

**2. Create a virtual environment and install packages**
```
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

**3. Get the dataset** (about 1.2 GB)
```
python download_dataset.py
```
If that fails, download it manually from
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia , unzip it, and make sure you end up with
`data/chest_xray/train/NORMAL`, `data/chest_xray/train/PNEUMONIA`, `data/chest_xray/test/NORMAL`, `data/chest_xray/test/PNEUMONIA`.

**4. Train the model**
```
python train.py
```
Takes ~10-20 min on a GPU, ~1-2 hours on CPU. Use `python train.py --epochs 4` for a quicker run.
Results (accuracy, recall, confusion matrix, graphs) appear in `outputs/`.

**5. Run the app**
```
python run.py
```
Open **http://127.0.0.1:8000** in your browser, upload an X-ray, and click Analyze.
(Interactive API docs: http://127.0.0.1:8000/docs)

## Testing without your own X-ray
Use any image from `data/chest_xray/test/NORMAL` or `.../PNEUMONIA`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Dataset not found` | Check the folder path in Step 3 |
| "Model not found" in the browser | Run `python train.py` first |
| `Cannot reach the server` | Make sure `python run.py` is running |
| Out of memory while training | `python train.py --batch_size 8 --workers 0` |
| `torch` install is slow / fails | See https://pytorch.org/get-started/locally |

## Ideas to extend
Other datasets (brain MRI, skin lesions, plant disease) - change `CLASSES` in `backend/model_utils.py`,
use another dataset folder, retrain. Add Docker, user login, or PDF reports.
