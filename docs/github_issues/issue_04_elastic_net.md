# Issue #4: Add Elastic Net Regularization

**Title:** Add Elastic Net regularization option

**Labels:** enhancement

**Milestone:** v1.1

## Description

Implement elastic net (combined L1+L2) regularization with l1_ratio parameter.

## Background

Elastic Net combines Ridge (L2) and Lasso (L1) regularization, providing the benefits of both: feature selection from Lasso and handling of correlated features from Ridge.

## Requirements

- Add elastic net penalty option to both linear and logistic regression
- Implement l1_ratio parameter (0 = Ridge, 1 = Lasso, 0.5 = equal mix)
- Maintain statistical inference capabilities where possible
- Optimize for computational efficiency

## Acceptance Criteria

- [ ] Add `penalty='elastic_net'` option to regression classes
- [ ] Add `l1_ratio` parameter (default 0.5)
- [ ] Implement proximal gradient descent for optimization
- [ ] Handle edge cases (l1_ratio=0 should equivalent to Ridge, l1_ratio=1 to Lasso)
- [ ] Maintain compatibility with existing regularization code
- [ ] Unit tests comparing to scikit-learn ElasticNet
- [ ] Performance benchmarks
- [ ] Documentation with examples

## Implementation Notes

```python
from pytorch_regression import LinearRegression

# Elastic net with equal L1/L2 mix
model = LinearRegression(penalty='elastic_net', alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)

# More L1 (Lasso-like)
model_sparse = LinearRegression(penalty='elastic_net', alpha=0.1, l1_ratio=0.8)
model_sparse.fit(X_train, y_train)
```

## Technical Details

- Objective: `loss + alpha * (l1_ratio * ||w||_1 + (1-l1_ratio) * ||w||_2^2)`
- Use proximal gradient descent or coordinate descent
- Implement soft-thresholding for L1 component
- Provide regularization path computation capability

## Priority

High - Popular regularization method, should be available in v1.1.