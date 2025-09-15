# Issue #6: Implement regularization path computation

**Title:** Implement regularization path computation

**Labels:** enhancement, performance

**Milestone:** v2.0

## Description

Add ability to compute full regularization paths for L1/L2 penalties, showing how coefficients change across different regularization strengths.

## Background

Regularization paths show how model coefficients change as the regularization parameter varies. This is crucial for hyperparameter selection and understanding model behavior.

## Requirements

- Compute coefficient paths efficiently across regularization values
- Support both L1 (Lasso path) and L2 (Ridge path) regularization
- Integrate with cross-validation for optimal alpha selection
- Provide visualization utilities

## Acceptance Criteria

- [ ] Add `regularization_path()` method to regression classes
- [ ] Efficient path computation (warm starts, early stopping)
- [ ] Return coefficients, alphas, and optionally CV scores
- [ ] Support both automatic alpha grid and user-specified values
- [ ] Add `plot_path()` method for visualization
- [ ] Cross-validation integration for alpha selection
- [ ] Unit tests comparing to scikit-learn path algorithms
- [ ] Performance benchmarks for large datasets

## Implementation Notes

```python
from pytorch_regression import LinearRegression

model = LinearRegression()

# Compute regularization path
alphas, coefs, cv_scores = model.regularization_path(
    X_train, y_train,
    penalty='l1',
    alphas=None,  # Auto-generate
    cv=5,
    n_alphas=100
)

# Find optimal alpha
optimal_idx = np.argmin(cv_scores.mean(axis=1))
optimal_alpha = alphas[optimal_idx]

# Visualize path
model.plot_path(alphas, coefs, optimal_alpha=optimal_alpha)

# Fit final model with optimal alpha
final_model = LinearRegression(penalty='l1', alpha=optimal_alpha)
final_model.fit(X_train, y_train)
```

## Technical Details

- Use warm starts for computational efficiency
- Implement coordinate descent for Lasso path
- Use Cholesky updates for Ridge path
- Support early stopping based on convergence
- Automatic alpha grid generation (log-spaced)

## Advanced Features

- [ ] Elastic net path computation
- [ ] Group Lasso path (for categorical variables)
- [ ] Stability path analysis
- [ ] Feature importance ranking across path

## Priority

Low - Advanced feature for v2.0, not needed for initial release.