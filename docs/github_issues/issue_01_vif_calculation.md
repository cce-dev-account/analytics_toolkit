# Issue #1: Add Variance Inflation Factor (VIF) calculation utility

**Title:** Add Variance Inflation Factor (VIF) calculation utility

**Labels:** enhancement, statistics

**Milestone:** v1.1

## Description

Implement a standalone utility function to calculate VIF for feature selection/multicollinearity detection before model fitting.

## Background

Variance Inflation Factor (VIF) is a critical statistic for detecting multicollinearity in regression models. VIF measures how much the variance of a regression coefficient increases due to collinearity.

## Requirements

- Standalone function that can be used independently of regression models
- Calculate VIF for each feature in a dataset
- Handle both continuous and categorical variables
- Return results in a pandas DataFrame with feature names and VIF values
- Warn when VIF > 5 (moderate multicollinearity) or VIF > 10 (high multicollinearity)

## Acceptance Criteria

- [ ] Function `calculate_vif(X, feature_names=None)` implemented
- [ ] Handles pandas DataFrames and numpy arrays
- [ ] Returns VIF values for all features
- [ ] Includes warnings for high VIF values
- [ ] Unit tests comparing results to statsmodels VIF calculations
- [ ] Documentation with examples

## Implementation Notes

```python
from pytorch_regression.utils import calculate_vif

# Example usage
vif_results = calculate_vif(X_data)
print(vif_results)
# Output:
#     feature      VIF
# 0   feature1    2.34
# 1   feature2    8.67  # Warning: moderate multicollinearity
# 2   feature3   12.45  # Warning: high multicollinearity
```

## Priority

Medium - Important for model diagnostics but not blocking for initial release.