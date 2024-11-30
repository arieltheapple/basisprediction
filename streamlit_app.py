import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title of the app
st.title("🌽Basis Prediction🏡")

# Step 1: Load the Dataset (Preloaded by Developer)
@st.cache
def load_data():
    # Replace 'rf_data.csv' with the path to your dataset
    data = pd.read_csv("rf_data.csv")  # Ensure the dataset is in the same folder or provide the full path
    return data

# Load the dataset
data = load_data()

# Predefined independent and dependent variables
original_independent_vars = [
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

# Step 2: Handle categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=["Board_Month"], drop_first=True)

# Update independent_vars to reflect the transformed column names
independent_vars = [col for col in data.columns if col in original_independent_vars or col.startswith("Board_Month")]

# Get one-hot encoded Board_Month columns
board_month_columns = data.filter(like="Board_Month").columns.tolist()

# Extract month names for dropdown (e.g., "Jul", "Sep")
month_names = [col.replace("Board_Month_", "") for col in board_month_columns]
month_names.insert(0, "Dec")  # Add "Dec" as an option

# Split data into predictors and target
X = data[independent_vars]
y = data[dependent_var]

# Train Random Forest model with new hyperparameters
best_rf_model = RandomForestRegressor(
    n_estimators=100,         # Best number of trees
    min_samples_split=2,      # Minimum samples required to split an internal node
    min_samples_leaf=1,       # Minimum samples required at a leaf node
    max_features='log2',      # Number of features considered for splitting
    max_depth=30,             # Maximum depth of the tree
    random_state=123          # For reproducibility
)

# Fit the model
best_rf_model.fit(X, y)

# Step 3: Input Fields for Prediction
st.write("### Enter Feature Values for Prediction")
user_input = {}

# Collect user inputs for independent variables
for feature in original_independent_vars:
    label = corresponding_words.get(feature, feature)  # Get the corresponding word for the feature
    if feature == "Board_Month":
        # Dropdown for Board_Month
        selected_month = st.selectbox(f"{label}:", options=month_names)

        # Initialize Board_Month columns to 0
        for col in board_month_columns:
            user_input[col] = 0

        # If "Dec" is selected, leave all Board_Month columns as 0
        if selected_month != "Dec":
            # Set the corresponding one-hot column to 1
            one_hot_col = f"Board_Month_{selected_month}"
            user_input[one_hot_col] = 1
    else:
        # Number input for numerical variables
        value = st.number_input(f"{label}:", value=0.0)
        user_input[feature] = value

# Step 4: Make Prediction
if st.button("Predict"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    prediction = best_rf_model.predict(input_df)
    st.success(f"The predicted value for {dependent_var} is: {prediction[0]:.2f}")

