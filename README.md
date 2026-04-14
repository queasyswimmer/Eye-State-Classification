# Eye-State Classification from EEG Data 

This project utilizes supervised machine learning to classify whether a subject's eyes are open or closed based on EEG scan data. This research was conducted as part of my math studies at Emory University.

## Project Overview

The goal of this project is to analyze **frequency-band power** extracted from EEG scans to predict cognitive states. By processing raw brainwave data into power spectral density across different frequency bands, we can train models to distinguish between "eyes open" and "eyes closed" states with high accuracy.

Raw EDF recordings are segmented into **2-second windows**, and for each window we compute **log band power** across four frequency bands using Welch's method:

| Band  | Frequency Range |
|-------|----------------|
| Delta | 1 – 4 Hz       |
| Theta | 4 – 8 Hz       |
| Alpha | 8 – 13 Hz      |
| Beta  | 13 – 30 Hz     |

With 64 EEG channels × 4 bands, each sample is represented as a **256-dimensional feature vector**. The final dataset contains **3,000 samples** (2,250 train / 750 test), perfectly balanced between the two classes.

## Repository Structure

The project is organized to separate the data exploration from the specific model implementations:

- **`notebooks/`**
  - `general_data.ipynb` — Data cleaning, preprocessing, and calculating frequency-band power.
  - `knn.ipynb` — Implementation of the **k-Nearest Neighbors (kNN)** algorithm.
  - `RF.ipynb` — Implementation of the **Random Forest (RF)** algorithm.

## Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-Learn, Matplotlib, MNE, SciPy, Seaborn
- **Environment:** Developed using Google Colab

## Models & Results

Two supervised machine learning models were evaluated and tuned with cross-validated hyperparameter search.

### k-Nearest Neighbors (kNN)

A `GridSearchCV` over `n_neighbors`, distance metric (`p`), and weighting scheme identified the optimal configuration:

| Hyperparameter | Best Value |
|----------------|-----------|
| `n_neighbors`  | 3          |
| Distance metric | Euclidean |
| Weights        | Uniform    |

### Random Forest (RF)

A `RandomizedSearchCV` over tree depth, number of estimators, and split criteria identified the optimal configuration:

| Hyperparameter      | Best Value |
|---------------------|-----------|
| `n_estimators`      | 400        |
| `max_depth`         | 60         |
| `min_samples_split` | 5          |
| `max_features`      | sqrt       |

### Accuracy Summary

| Model                     | CV Accuracy | Test Accuracy |
|---------------------------|-------------|---------------|
| k-Nearest Neighbors (kNN) | 86.98%      | **87.47%**    |
| Random Forest (RF)        | 84.76%      | **87.60%**    |

Both models achieve similar test accuracy (~87.5%), with Random Forest edging ahead slightly. The alpha band (8–13 Hz) — known to increase in power when eyes are closed — is among the most predictive features, consistent with established neuroscience literature.

## Installation & Usage

To run these notebooks locally or in an environment like Google Colab, install the following dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib mne scipy seaborn
```

The notebooks expect EEG data in `.edf` format organized as:

```
eeg_eyes_project/
├── S001R01.edf   # Eyes open
├── S001R02.edf   # Eyes closed
├── S002R01.edf
├── S002R02.edf
└── ...
```

Update the `DATA_DIR` variable at the top of each notebook to point to your data folder.
