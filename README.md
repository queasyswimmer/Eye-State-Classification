# Eye-State Classification from EEG Data 🧠

This project utilizes supervised machine learning to classify whether a subject's eyes are open or closed based on EEG scan data. This research was conducted as part of my Applied Math studies at Emory University.

## 📊 Project Overview
The goal of this project is to analyze **frequency-band power** extracted from EEG scans to predict cognitive states. By processing raw brainwave data into power spectral density across different frequency bands, we can train models to distinguish between "eyes open" and "eyes closed" states with high accuracy.

## 📂 Repository Structure
The project is organized to separate the data exploration from the specific model implementations:

* **`notebooks/`**: 
    * `general_data.ipynb`: Data cleaning, preprocessing, and calculating frequency-band power.
    * `knn.ipynb`: Implementation of the **k-Nearest Neighbors (kNN)** algorithm.
    * `RF.ipynb`: Implementation of the **Random Forest (RF)** algorithm.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NumPy, Pandas, Scikit-Learn, Matplotlib
* **Environment:** Developed using Google Colab

## 🚀 Models & Results
I evaluated two primary supervised machine learning models: **k-Nearest Neighbors** and **Random Forest**. These models were trained on the processed EEG features to compare their predictive performance.

| Model | Classification Accuracy |
| :--- | :--- |
| **k-Nearest Neighbors (kNN)** | [Insert % from knn.ipynb] |
| **Random Forest (RF)** | [Insert % from RF.ipynb] |

## ⚙️ Installation & Usage
To run these notebooks locally or in an environment like Google Colab, you will need to install the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
