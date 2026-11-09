# Parkinson's Speech + Age + Sex Classifier

This is the updated version of the existing Streamlit app, using the notebook's saved `Speech + Age + Sex` model.

## Inputs
- 80 precomputed speech features
- Age
- Sex (`0 = male`, `1 = female` in uploaded files)

Total: 82 inputs. The app's downloadable CSV template uses the exact column names expected by the saved model.

## Run on Windows
1. Install Python 3.11 or 3.12.
2. Open PowerShell in this folder.
3. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
4. Install dependencies and start the app:
   ```powershell
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```
5. Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Important
- The saved model was created with scikit-learn 1.6.1, so requirements pins that version for compatibility.
- The app uses precomputed speech features; it does not extract features from raw audio.
- This is a research/classroom prototype, not a medical diagnostic tool.
