import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    """
    Creates new features from the existing ones to improve model performance.
    This includes nutrient ratios, total nutrient content, and interaction features.
    """
    df = df.copy()
    
    # A small constant (epsilon) is added to denominators to prevent division by zero errors.
    epsilon = 1e-6
    
    # Nutrient Ratios 
    # balance between nutrients is often more important than absolute values.
    df['N_P_Ratio'] = df['Nitrogen (N)'] / (df['Phosphorus (P)'] + epsilon)
    df['N_K_Ratio'] = df['Nitrogen (N)'] / (df['Potassium (K)'] + epsilon)
    df['P_K_Ratio'] = df['Phosphorus (P)'] / (df['Potassium (K)'] + epsilon)
    
    # Total Nutrient Content
    # The overall concentration of nutrients.
    df['Total_Nutrients'] = df['Nitrogen (N)'] + df['Phosphorus (P)'] + df['Potassium (K)']
    
    # Polynomial Features
    # Helps the model capture non-linear relationships.
    df['Temp_sq'] = df['Temperature'] ** 2
    df['Moisture_sq'] = df['Moisture Content'] ** 2
    
    # --- Interaction Features ---
    # Captures how two variables work together.
    df['Temp_x_Moisture'] = df['Temperature'] * df['Moisture Content']
    
    return df

# Load All Data 
print("Loading data...")
try:
    TRAIN_PATH = '/kaggle/input/playground-series-s5e6/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e6/test.csv'
    EXTERNAL_CSV_PATH = '/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv'
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    external_df = pd.read_csv(EXTERNAL_CSV_PATH)
    
    print("All data files loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Could not find competition data files. Please ensure the paths are correct.")
    exit()

# Preprocessing and Feature Engineering

# Store test IDs for the final submission and drop the 'id' column from the dataframes.
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Standardize column names across all dataframes.
column_rename_map = {
    'Temparature': 'Temperature',
    'Moisture': 'Moisture Content',
    'Soil Type': 'Soil Color',
    'Crop Type': 'Crop',
    'Nitrogen': 'Nitrogen (N)',
    'Potassium': 'Potassium (K)',
    'Phosphorous': 'Phosphorus (P)'
}

train_df.rename(columns=column_rename_map, inplace=True, errors='ignore')
test_df.rename(columns=column_rename_map, inplace=True, errors='ignore') 
external_df.rename(columns=column_rename_map, inplace=True, errors='ignore')
print("Standardized column names.")

# Combine the competition & external training data.
# Column names are aligned to ensure compatibility.
train_df = pd.concat([train_df, external_df[train_df.columns]], ignore_index=True)
print("Combined external data with training data.")


# Apply the feature engineering function to both training and test sets.
print("Applying feature engineering...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# Define the target column and separate features (X) from the target (y).
TARGET_COL = 'Fertilizer Name'
X = train_df.drop(TARGET_COL, axis=1)
y_raw = train_df[TARGET_COL]

# Ensure the test set has the exact same columns as the training feature set.
test_df = test_df[X.columns]

# Encode the categorical TARGET variable
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Encode CATEGORICAL FEATURES using one-hot encoding.
# combine X and test_df 
combined_features = pd.concat([X, test_df], ignore_index=True)
combined_features_encoded = pd.get_dummies(combined_features, columns=['Soil Color', 'Crop'], drop_first=True)

# Separate the combined dataframe back into final training and testing sets.
X_final = combined_features_encoded.iloc[:len(train_df)]
X_test_final = combined_features_encoded.iloc[len(train_df):]

print(f"Preprocessing complete. Final training features shape: {X_final.shape}")
print(f"Final testing features shape: {X_test_final.shape}")

# Model Evaluation (using a validation set)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

print("Splitting data for validation...")
# Split the final training data into a training and validation set with a 80/20 split.
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y,
    test_size=0.2,    
    random_state=42,  # Ensures the split is the same every time
    stratify=y        # Ensures the distribution of target classes is the same in train and validation sets
)

print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples.")

print("Training a temporary model for validation...")
# Initialize and train the model on the new, smaller training set
validation_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
validation_model.fit(X_train, y_train)

print("Evaluating model on the validation set...")
# Make predictions on the validation set
val_preds = validation_model.predict(X_val)

# Calculate the F1 score.
f1 = f1_score(y_val, val_preds, average='weighted')

print(f"\nValidation F1 Score: {f1:.4f}\n")

print("Training the Random Forest model on the full featured data...")

# Model training
print("Training the Random Forest model on the featured data...")
rf_model = RandomForestClassifier(
    n_estimators=200,         # Number of trees in the forest.
    max_depth=10,             # Maximum depth of the trees to prevent overfitting.
    min_samples_leaf=5,       # Minimum samples required at a leaf node.
    random_state=42,          # For reproducibility.
    n_jobs=-1                 # Use all available CPU cores.
)
# Train the model on the final data with engineered features.
rf_model.fit(X_final, y)
print("Model training complete.")


# Prediction
print("Generating predictions on the final test set...")
# Predict probabilities for each class on the test data.
test_probabilities = rf_model.predict_proba(X_test_final)

# Get the indices of the top 3 predictions for each test sample.
# np.argsort sorts in ascending order, so we take the last 3 columns.
top3_indices = np.argsort(test_probabilities, axis=1)[:, -3:]

# Decode the predictions from integer labels back to the original fertilizer names.
top3_names = le.inverse_transform(top3_indices.flatten()).reshape(top3_indices.shape)

# Format the predictions by joining the three fertilizer names with a space.
predictions = [' '.join(row) for row in top3_names]
print("Top 3 predictions generated.")


# Submission File
print("Creating submission file...")
submission_df = pd.DataFrame({'id': test_ids, 'Fertilizer Name': predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Final check on submission format:")
print(submission_df.head())