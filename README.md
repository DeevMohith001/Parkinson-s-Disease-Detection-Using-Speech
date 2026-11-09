
# Parkinson's Disease Detection Using Speech Features

A Machine Learning-based research project that explores the detection of Parkinson's Disease using speech-related acoustic features and demographic information.


## Project Overview

Parkinson's Disease is a progressive neurological disorder that affects movement and can also influence speech patterns.

This project investigates whether acoustic speech characteristics, combined with demographic information, can help distinguish between healthy individuals and individuals with Parkinson's Disease.

### Input Features

The model uses 82 numerical input features:

| Feature Category | Number of Features |
|---|---:|
| LPC | 20 |
| LAR | 20 |
| Cepstral Features | 20 |
| MFCC | 20 |
| Age | 1 |
| Sex | 1 |
| **Total** | **82** |

### Prediction Classes

- `0` - Healthy Control (HC)
- `1` - Parkinson's Disease (PwPD)

Sex encoding:
- `0` - Male
- `1` - Female

## Features

- Machine Learning-based Parkinson's Disease classification.
- KNN classification model.
- Interactive Streamlit web interface.
- Manual entry of speech and demographic features.
- CSV and Excel file upload support.
- Model-estimated probability of Parkinson's Disease.
- Downloadable input template.
- Jupyter notebooks for exploratory analysis and experimentation.

## Tech Stack

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib
- **Development Environment:** Jupyter Notebook

## Project Structure

```text
Parkinson-Disease-Detection/
│
├── notebooks/
│   ├── parkinson_speech_model.ipynb
│   └── speech_feature_analysis.ipynb
│
├── parkinson_speech_app/
│   ├── app.py
│   ├── predict.py
│   ├── train_compare.py
│   ├── feature_names.json
│   └── notebook_model.joblib
│
├── test_datapoints/
│   ├── disease/
│   ├── healthy/
│   └── parkinson_speech_age_sex_template.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Parkinson-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run parkinson_speech_app/app.py
```


## How to Use

1. Launch the Streamlit application.
2. Enter the required speech features and demographic information manually, or upload a CSV/XLSX file.
3. Ensure that all 82 required input features are provided.
4. Click the Predict button.
5. View the predicted class and model-estimated probability.

**Note:** The application currently accepts precomputed speech features. It does not directly extract features from raw WAV audio recordings.


## Machine Learning Models

Multiple Machine Learning algorithms were implemented and evaluated to identify a suitable classifier for Parkinson's Disease detection.

### Models Evaluated

The following classification algorithms were experimented with:

1. Logistic Regression
2. Support Vector Machine (SVM - RBF Kernel)
3. Support Vector Machine (SVM - Linear Kernel)
4. K-Nearest Neighbors (KNN)
5. Random Forest
6. Extra Trees Classifier
7. Linear Discriminant Analysis (LDA)

### Model Evaluation and Selection

- Baseline model comparison was performed using Stratified 5-Fold Cross-Validation.
- Hyperparameter optimization was performed using GridSearchCV.
- Balanced Accuracy was used as the primary optimization metric in the tuning experiments.
- Different feature selection and preprocessing configurations were explored.
- Nested Cross-Validation was also performed to obtain a more robust estimate of model performance.

### Final Selected Model

After experimentation and hyperparameter tuning, the final model saved for the Streamlit application was:

**K-Nearest Neighbors (KNN)**

The final model uses:
- 3 nearest neighbors
- Euclidean distance
- Uniform neighbor weights
- Speech features combined with Age and Sex

The trained preprocessing and classification pipeline is serialized using Joblib and loaded by the Streamlit application for predictions.

### Model Performance

Model performance was evaluated using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

The final experimental notebook also includes out-of-fold evaluation and confusion matrix analysis.


## Future Improvements

- Implement automatic speech feature extraction from raw audio.
- Experiment with additional Machine Learning classifiers.
- Improve model generalization through participant-level validation.
- Explore Deep Learning approaches for speech analysis.
- Develop a more comprehensive prediction interface.

## Disclaimer

This project is developed for educational and research purposes only.

It is not a clinically validated medical diagnostic system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.
