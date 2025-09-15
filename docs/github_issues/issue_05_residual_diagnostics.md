# Issue #5: Add comprehensive residual diagnostic methods

**Title:** Add comprehensive residual diagnostic methods

**Labels:** enhancement, statistics

**Milestone:** v1.3

## Description

Implement leverage statistics, Cook's distance, and other influence measures for comprehensive model diagnostics.

## Background

Residual diagnostics are essential for validating regression assumptions and identifying influential observations that may disproportionately affect model results.

## Requirements

- Implement multiple diagnostic measures
- Provide easy-to-use interface for model diagnostics
- Include visualization helpers
- Handle both linear and logistic regression

## Subtasks

- [ ] Standardized residuals
- [ ] Leverage (hat matrix diagonal)
- [ ] Cook's distance
- [ ] DFFITS (Difference in Fits)
- [ ] Studentized residuals
- [ ] Outlier detection methods

## Acceptance Criteria

- [ ] Add `diagnostics()` method to regression classes
- [ ] Return diagnostic measures in pandas DataFrame
- [ ] Include interpretation thresholds (e.g., Cook's D > 4/n)
- [ ] Add plotting methods for diagnostic visualizations
- [ ] Handle large datasets efficiently (avoid computing full hat matrix)
- [ ] Unit tests comparing to established statistical software
- [ ] Documentation with interpretation guidelines

## Implementation Notes

```python
from pytorch_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Get comprehensive diagnostics
diagnostics = model.diagnostics(X_train, y_train)
print(diagnostics.head())
#    observation  residual  standardized_residual  leverage  cooks_distance  dffits
# 0            0     -0.23                  -0.45      0.12            0.003   -0.15
# 1            1      1.45                   2.89      0.08            0.067    0.82
# 2            2      0.67                   1.33      0.15            0.027    0.52

# Identify influential observations
influential = diagnostics[diagnostics['cooks_distance'] > 4/len(X_train)]
print(f"Found {len(influential)} influential observations")

# Generate diagnostic plots
model.plot_diagnostics(X_train, y_train)
```

## Technical Details

- Leverage: h_i = x_i^T * (X^T*X)^(-1) * x_i
- Cook's Distance: D_i = (e_i^2 / (p*MSE)) * (h_i / (1-h_i)^2)
- DFFITS: DFFITS_i = t_i * sqrt(h_i / (1-h_i))
- Use efficient algorithms for large datasets (approximate leverages if needed)

## Priority

Medium - Important for model validation but not essential for basic functionality.