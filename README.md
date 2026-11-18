# Bi-LSTM-BP

Cuffless blood pressure estimation: predicts **systolic (SBP)** and **diastolic (DBP)** blood pressure from **PPG** signal features using stacked **Bidirectional LSTM** regression models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AchmadRifqyPratama/Bi-LSTM-BP/blob/main/Bi_LSTM_BP.ipynb)

## Stack

Python, TensorFlow / Keras, scikit-learn, SciPy, NumPy, pandas, Matplotlib. Written for Google Colab with data on Google Drive.

## Pipeline (`Bi_LSTM_BP.ipynb`)

1. **Data loading**: 500 records `rec_1.csv` ... `rec_500.csv` from a MIMIC-derived dataset (`dataset_MIMIC/Samples`). Each record holds PPG, ABP and ECG channels sampled at 125 Hz.
2. **Preprocessing**: PPG is band-pass filtered (Chebyshev type II, 0.5–5 Hz) and min-max normalized. Signals are segmented into 10 s windows (1250 samples) with 5 s overlap.
3. **Feature extraction (per window)**:
   - PPG inputs: signal energy, median systolic peak interval, median diastolic peak interval, median systolic peak amplitude, median diastolic peak amplitude.
   - Targets: median SBP and DBP taken from ABP peaks and troughs.
   - Windows with missing or mismatched peaks are discarded.
4. **Split and scaling**: 90/10 train/test split (`random_state=0`) and `StandardScaler` on the inputs.
5. **Model**: `create_model_Bi()` stacks 3 `Bidirectional(LSTM(256))` layers, each followed by Dropout 0.2, then `Dense(1)`. Loss is MAE, optimizer is Adam, trained for 500 epochs. SBP and DBP use separate models.
6. **Evaluation**: train/test MAE, predicted-vs-reference plots and Pearson correlation for SBP and DBP. Models are saved as `.h5` files under `Result_SBP/` and `Result_DBP/` on Drive.

## How to run

1. Open the notebook in Colab (badge above) and mount Google Drive.
2. Place the dataset at `<working_dir>/dataset_MIMIC/Samples/rec_N.csv` and set `working_dir` in the first cell.
3. Run all cells in order.

The dataset and trained models are not included in this repository.
