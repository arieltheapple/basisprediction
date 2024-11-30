import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Title of the app
st.title("🌽Basis Prediction🏡")

# Step 1: Load the Dataset (Preloaded by Developer)
@st.cache
def load_data():
    # Replace 'your_data.csv' with the path to your dataset
    data = pd.read_csv("rf_data.csv")  # Ensure the dataset is in the same folder or provide the full path
    return data

# Load the dataset
data = load_data()

# Predefined independent and dependent variables
independent_vars = [
    "Distance", "Weekly_NY_Upstate_Gasoline_Price", "VIX",
    "Total_Silage_P", "Volume", "Wyne_Basis", "Contract_Duration",
    "Engaging_Period", "Lag_Monthly_NY_Gasoline_Avg_P", "Board_Month",
    "future_Price", "Lag_Monthly_Wyne_Avg_B"
]
dependent_var = "Basis"

corresponding_words = {
    "Distance": "Driving Distance",
    "Weekly_NY_Upstate_Gasoline_Price": "Weekly Average NY Upstate Gasoline Price",
    "VIX": "VIX",
    "Total_Silage_P": "Western NY Yearly Silage Production",
    "Volume": "Buying Volume (Bushels)",
    "Wyne_Basis": "WYNE Basis",
    "Contract_Duration": "Contract Duration",
    "Engaging_Period": "Engaging Period",
    "Lag_Monthly_NY_Gasoline_Avg_P": "Last Month's Average NY Upstate Gasoline Price",
    "Board_Month": "Board Month",
    "future_Price": "Future Price",
    "Lag_Monthly_Wyne_Avg_B": "Last Month's Average WYNE Basis"
}

# Encode categorical variables (e.g., "Board_Month")
categorical_vars = ["Board_Month"]
label_encoder = LabelEncoder()
data["Board_Month"] = label_encoder.fit_transform(data["Board_Month"])

# Prepare data for training
X = data[independent_vars]
y = data[dependent_var]

# Step 2: Train the Random Forest Model with Best Hyperparameters
best_model = RandomForestRegressor(
    n_estimators=500,         # Best number of trees
    max_features=7,         # Number of features considered for splitting
    random_state=42,           # For reproducibility
    n_jobs=-1                 # Use all cores for computation
)

# Fit the model
best_model.fit(X, y)

# Step 3: Input Fields for Prediction
st.write("### Enter Feature Values for Prediction")
user_input = {}

# Collect user inputs for independent variables
for feature in independent_vars:
    label = corresponding_words[feature]  # Get the corresponding word for the feature
    if feature in categorical_vars:
        # Dropdown for categorical variables
        value = st.selectbox(f"{label}:", options=label_encoder.classes_)
        user_input[feature] = label_encoder.transform([value])[0]
    else:
        # Number input for numerical variables
        value = st.number_input(f"{label}:", value=0.0)
        user_input[feature] = value

# Step 4: Make Prediction
if st.button("Predict"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    prediction = best_model.predict(input_df)
    st.success(f"The predicted value for {dependent_var} is: {prediction[0]:.2f}")
