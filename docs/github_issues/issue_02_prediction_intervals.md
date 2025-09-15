# Issue #2: Add prediction intervals for LinearRegression

**Title:** Add prediction intervals for LinearRegression

**Labels:** enhancement, statistics

**Milestone:** v1.2

## Description

Implement prediction intervals (not just confidence intervals) for individual predictions in linear regression models.

## Background

Prediction intervals provide bounds for individual future observations, accounting for both parameter uncertainty and residual variance. This is different from confidence intervals which only account for parameter uncertainty.

## Requirements

- Calculate prediction variance correctly for new observations
- Handle both single and multiple predictions efficiently
- Add `predict_interval()` method to LinearRegression class
- Support different confidence levels (default 95%)

## Acceptance Criteria

- [ ] Calculate prediction variance correctly: Var(y_new) = σ² * (1 + x_new^T * (X^T*X)^(-1) * x_new)
- [ ] Handle both single and multiple predictions efficiently
- [ ] Add `predict_interval(X, alpha=0.05)` method
- [ ] Return tuple of (predictions, lower_bounds, upper_bounds)
- [ ] Use appropriate t-distribution for small samples
- [ ] Unit tests comparing to manual calculations
- [ ] Documentation with examples

## Implementation Notes

```python
from pytorch_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Get prediction intervals
predictions, lower, upper = model.predict_interval(X_test, alpha=0.05)

# For visualization
import matplotlib.pyplot as plt
plt.fill_between(range(len(predictions)), lower, upper, alpha=0.3, label='95% Prediction Interval')
plt.plot(predictions, label='Predictions')
plt.legend()
```

## Technical Details

- Use residual standard error from model fit
- Calculate leverages for new observations
- Account for degrees of freedom in t-distribution
- Optimize for batch predictions using vectorized operations

## Priority

High - Critical for statistical inference and uncertainty quantification.