# MicroMorph AI - Backend

MicroMorph is an intelligent embedded microscopy platform designed to identify, classify, and count marine micro-organisms in real-time. This repository contains the backend processing engine, which leverages a multi-model ensemble to analyze microscopic images and provide ecological insights.

## 🚀 Features
* [cite_start]**Multi-Model Ensemble:** Combines five different perspectives for species-level recognition, including visual detection and morphological analysis.
* [cite_start]**Feature Extraction:** Computes 131+ biological parameters (geometric, Hu, Zernike, color, and texture features) directly from organism contours.
* [cite_start]**Similarity Learning:** Uses embedding-based learning to identify and store unseen organisms for continuous system improvement.
* [cite_start]**Statistical Reliability:** Includes built-in metrics to calculate mean confidence, standard deviation, and reliability scores for model predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, Torchvision (Mask R-CNN), Ultralytics (YOLO) 
* **Computer Vision:** OpenCV, Scikit-image, Mahotas
* **Vector Database:** ChromaDB 
* **Other Libraries:** NumPy, Pandas, Joblib, tqdm 

## 📁 Key Components
* `ModelSync/Metrics/MaskRCNN.py`: Handles Mask R-CNN inference and reliability statistics.
* `ModelSync/Metrics/Polygons/Extract.py`: Extracts 131+ morphological features from COCO-format polygons.
* `ModelSync/Metrics/embeds.py`: Manages image embeddings and ChromaDB queries for species identification.
* `ModelSync/Metrics/random_forest.py`: Integrates YOLO segmentation with Random Forest classification.

## ⚙️ Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/the-madhankumar/MicroMorphBackend](https://github.com/the-madhankumar/MicroMorphBackend)
Install dependencies:Ensure you have the required Python libraries installed (OpenCV, PyTorch, ChromaDB, etc.).Model Paths:
Update the local paths for MASK_R_CNN_PATH, RANDOMFOREST_MODEL, and YOLOSEG_RF in the respective script files to match your local environment.
