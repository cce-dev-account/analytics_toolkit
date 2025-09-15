# Issue #3: Target-based Categorical Encoding

**Title:** Implement target-based encoding for categorical variables

**Labels:** enhancement, feature

**Milestone:** v1.2

## Description

Add option for target-based encoding as alternative to dummy variables, with proper cross-validation to prevent leakage.

## Background

Target-based encoding (mean encoding) replaces categorical values with the mean of the target variable for that category. This can be more effective than dummy variables for high-cardinality categorical features, but requires careful handling to prevent overfitting.

## Requirements

- Target-based encoding option in regression classes
- Cross-validation to prevent target leakage
- Smoothing techniques for categories with few observations
- Fallback to global mean for unseen categories during prediction

## Acceptance Criteria

- [ ] Add `categorical_encoding='dummy'|'target'` parameter to regression classes
- [ ] Implement k-fold cross-validation for target encoding
- [ ] Add smoothing parameter for regularization
- [ ] Handle unseen categories during prediction
- [ ] Preserve encoding mappings for consistent predictions
- [ ] Unit tests comparing performance vs dummy encoding
- [ ] Documentation with examples and warnings about overfitting

## Implementation Notes

```python
from pytorch_regression import LinearRegression

# Use target encoding instead of dummy variables
model = LinearRegression(categorical_encoding='target', cv_folds=5, smoothing=10)
model.fit(X_train, y_train)  # X_train contains categorical columns

# Encoding is automatically applied and preserved for prediction
y_pred = model.predict(X_test)
```

## Technical Details

- Use k-fold cross-validation during training
- Smoothing: `encoded_value = (count * category_mean + smoothing * global_mean) / (count + smoothing)`
- Store encoding mappings for consistent prediction
- Warn about potential overfitting risks

## Priority

Medium - Useful feature but not essential for core functionality.