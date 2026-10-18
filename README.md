# Employee Attrition Prediction

A machine learning project that explores employee attrition and estimates attrition risk with a trained Logistic Regression model. The project includes an analysis and training notebook, saved model artifacts, and a Streamlit application.

## Project structure

```text
.
├── app.py                              # Streamlit prediction app
├── dataset/
│   └── employee_attrition.csv          # Input dataset
├── models/
│   ├── feature_columns.pkl             # Feature order used during training
│   ├── logistic_regression_model.pkl   # Trained classifier
│   └── scaler.pkl                       # Fitted feature scaler
├── notebook/
│   └── Employee_Attrition.ipynb         # EDA, preprocessing, training, evaluation
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- `pip`

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate       # Windows
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

From the project root, start Streamlit:

```bash
python -m streamlit run app.py
```

Open the local address printed in the terminal (usually <http://localhost:8501>). Keep the terminal process running while using the app; press **Ctrl+C** there to stop it.

The app asks for a small set of influential employee attributes. Other model inputs are filled with typical values from the provided dataset. The prediction is an estimate from this model, not a determination about an individual employee.

## Explore and retrain

Open `notebook/Employee_Attrition.ipynb` in Jupyter and run its cells from top to bottom. The notebook covers data inspection, exploratory analysis, preprocessing, model training, and evaluation. Its final cell saves the classifier, scaler, and feature list to `models/`, which the app loads at startup.

If you retrain the model, rerun the final save cell before starting the app so all three artifacts match the new model.

## Model evaluation

The notebook's recorded Logistic Regression test results are:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8776 |
| Precision | 0.6571 |
| Recall | 0.4894 |
| F1 score | 0.5610 |
| ROC-AUC | 0.8033 |

The dataset is imbalanced, so accuracy alone does not describe how well the model identifies employees who leave. Review recall and the other metrics when interpreting results.

## Troubleshooting

- **`ModuleNotFoundError`**: Activate the project environment and run `python -m pip install -r requirements.txt`.
- **Model files not found**: Confirm the three `.pkl` files are in `models/`. Run the notebook through its final save cell to create them.
- **XGBoost reports missing `libomp.dylib` on macOS**: This affects the optional XGBoost notebook section, not the Streamlit app. You can skip that section if you only need the saved Logistic Regression model and app.
