# Fertilizer Recommendation System

A machine learning approach to predict optimal fertilizer recommendations based on soil conditions and crop requirements using Random Forest classification.

## Project Overview

This project tackles fertilizer recommendation as a multi-class classification problem, predicting the top 3 most suitable fertilizer types for given agricultural conditions. The model analyzes soil nutrient levels, environmental factors, and crop characteristics to generate recommendations.

## Dataset

The project combines multiple data sources:
- **Training Data**: 850,000 samples (including external fertilizer dataset)
- **Test Data**: 250,000 samples
- **Features**: 27 engineered features derived from 7 base features

### Input Features
- **Soil Nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
- **Environmental**: Temperature, Moisture Content
- **Categorical**: Soil Type, Crop Type

## Feature Engineering

The model implements several feature engineering techniques to capture agricultural relationships:

### Nutrient Ratios
- N/P, N/K, P/K ratios (nutrient balance is often more critical than absolute values)
- Total nutrient content

### Environmental Interactions
- Temperature² and Moisture² (polynomial features for non-linear relationships)
- Temperature × Moisture interaction terms

### Categorical Encoding
- One-hot encoding for soil types and crop varieties
- Ensures consistent feature space across train/test sets

## Model Architecture

**Algorithm**: Random Forest Classifier
- 200 estimators
- Max depth: 10 (prevents overfitting)
- Min samples per leaf: 5
- Stratified train/validation split (80/20)

## Results

- **Validation F1 Score**: 0.1437 (weighted average)
- **Output Format**: Top 3 fertilizer recommendations per sample
- **Model Performance**: Indicates significant room for improvement

## Key Challenges

1. **Multi-class complexity**: Predicting specific fertilizer products rather than nutrient requirements
2. **Feature scaling**: Mixed units across temperature, percentages, and ratios
3. **Domain knowledge integration**: Agricultural expertise needed for proper feature selection
4. **Evaluation metrics**: Gap between F1 scoring and top-3 prediction format

## Technical Implementation

The pipeline includes:
- Data standardization and cleaning
- Feature engineering with agricultural domain knowledge
- Model validation with stratified sampling
- Top-3 prediction generation using probability rankings

## Future Improvements

- Implement proper feature scaling and normalization
- Integrate soil pH and micronutrient data
- Add cross-validation for more robust evaluation
- Consider ensemble methods beyond Random Forest
- Incorporate seasonal and regional agricultural factors

## Usage

```python
# Load and preprocess data
train_df = create_features(train_df)
test_df = create_features(test_df)

# Train model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5)
rf_model.fit(X_final, y)

# Generate predictions
predictions = rf_model.predict_proba(X_test_final)
```

## Dependencies

- pandas
- numpy
- scikit-learn
- Standard Python data science stack

---

*Note: This is a competition-style implementation focused on prediction accuracy. Real-world agricultural applications would require additional domain expertise and validation.*
