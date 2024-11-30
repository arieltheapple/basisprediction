import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Step 2: Handle categorical variables using one-hot encoding
if not all(data[independent_vars].dtypes.apply(pd.api.types.is_numeric_dtype)):
    data = pd.get_dummies(data, columns=["Board_Month"], drop_first=True)

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
for feature in independent_vars:
    label = corresponding_words.get(feature, feature)  # Get the corresponding word for the feature
    if feature == "Board_Month":
        # Dropdown for categorical variables (Board_Month)
        value = st.selectbox(f"{label}:", options=data.filter(like="Board_Month").columns.tolist())
        # Set all Board_Month columns to 0 and set the selected one to 1
        for col in data.filter(like="Board_Month").columns:
            user_input[col] = 1 if col == value else 0
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
