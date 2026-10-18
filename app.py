from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "dataset" / "employee_attrition.csv"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👩‍💼",
    layout="wide",
)
st.title("Employee Attrition Prediction System")
st.write("Enter the employee information to estimate attrition risk.")

missing_files = [
    path for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATASET_PATH)
    if not path.is_file()
]
if missing_files:
    st.error(
        "Required project files are missing: "
        + ", ".join(str(path.relative_to(BASE_DIR)) for path in missing_files)
        + ". Run the notebook cells in order to create the model files."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
training_columns = joblib.load(FEATURES_PATH)
dataset = pd.read_csv(DATASET_PATH)

# Match the notebook preprocessing: discard identifier/constant columns and
# the target, then one-hot encode the remaining categorical features.
excluded_columns = {
    "Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"
}
input_columns = [column for column in dataset.columns if column not in excluded_columns]
categorical_columns = [
    column for column in input_columns
    if not pd.api.types.is_numeric_dtype(dataset[column])
]
useful_numeric_fields = [
    "Age",
    "YearsSinceLastPromotion",
    "TotalWorkingYears",
    "NumCompaniesWorked",
    "StockOptionLevel",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobLevel",
]
useful_categorical_fields = [
    "OverTime",
    "JobRole",
    "BusinessTravel",
    "MaritalStatus",
]
useful_fields = useful_numeric_fields + useful_categorical_fields
defaults = {
    column: (
        dataset[column].mode(dropna=True).iloc[0]
        if column in categorical_columns
        else int(dataset[column].median())
    )
    for column in input_columns
}

st.header("Employee Information")
st.caption(
    "Enter a few influential factors. Other model features use typical values "
    "from the training dataset."
)
with st.form("employee_information"):
    values = {}
    left, right = st.columns(2)
    for index, column in enumerate(useful_fields):
        container = left if index % 2 == 0 else right
        with container:
            if column in categorical_columns:
                options = sorted(dataset[column].dropna().unique().tolist())
                default_index = options.index(defaults[column])
                values[column] = st.selectbox(
                    column, options, index=default_index
                )
            else:
                series = dataset[column]
                values[column] = st.number_input(
                    column,
                    min_value=int(series.min()),
                    max_value=int(series.max()),
                    value=defaults[column],
                    step=1,
                )
    submitted = st.form_submit_button("Predict Attrition")

if submitted:
    complete_values = defaults.copy()
    complete_values.update(values)
    input_data = pd.DataFrame([complete_values], columns=input_columns)
    input_data = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    scaled_input = scaler.transform(input_data)
    prediction = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0, 1])

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High attrition risk · estimated probability: {probability:.1%}")
    else:
        st.success(f"Low attrition risk · estimated probability: {probability:.1%}")
