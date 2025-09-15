"""
Statistical computation functions for PyTorch regression module.
"""


import numpy as np
import torch
from scipy import stats


def compute_standard_errors(
    covariance_matrix: torch.Tensor, use_torch: bool = False
) -> torch.Tensor:
    """
    Compute standard errors from covariance matrix.

    Parameters
    ----------
    covariance_matrix : torch.Tensor
        Parameter covariance matrix.
    use_torch : bool, default=False
        Whether to use PyTorch for computation.

    Returns
    -------
    std_errors : torch.Tensor
        Standard errors.
    """
    if use_torch:
        return torch.sqrt(torch.diag(covariance_matrix))
    else:
        # Use numpy for numerical stability
        cov_np = covariance_matrix.detach().cpu().numpy()
        std_errors_np = np.sqrt(np.diag(cov_np))
        return torch.tensor(std_errors_np, device=covariance_matrix.device)


def compute_test_statistics(
    coef: torch.Tensor, std_errors: torch.Tensor, distribution: str = "t"
) -> torch.Tensor:
    """
    Compute test statistics (t or z statistics).

    Parameters
    ----------
    coef : torch.Tensor
        Model coefficients.
    std_errors : torch.Tensor
        Standard errors.
    distribution : str, default='t'
        Distribution type ('t' or 'z').

    Returns
    -------
    test_stats : torch.Tensor
        Test statistics.
    """
    # Avoid division by zero
    std_errors_safe = torch.where(std_errors > 1e-10, std_errors, torch.tensor(1e-10))
    return coef / std_errors_safe


def compute_p_values(
    test_stats: torch.Tensor, dof: int, distribution: str = "t"
) -> np.ndarray:
    """
    Compute p-values from test statistics.

    Parameters
    ----------
    test_stats : torch.Tensor
        Test statistics.
    dof : int
        Degrees of freedom.
    distribution : str, default='t'
        Distribution type ('t' or 'z').

    Returns
    -------
    p_values : np.ndarray
        Two-tailed p-values.
    """
    test_stats_np = test_stats.detach().cpu().numpy()

    if distribution == "t":
        # Two-tailed t-test
        p_values = 2 * (1 - stats.t.cdf(np.abs(test_stats_np), df=dof))
    elif distribution == "z":
        # Two-tailed z-test
        p_values = 2 * (1 - stats.norm.cdf(np.abs(test_stats_np)))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return p_values


def compute_confidence_intervals(
    coef: torch.Tensor,
    std_errors: torch.Tensor,
    alpha: float = 0.05,
    dof: int = None,
    distribution: str = "t",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for coefficients.

    Parameters
    ----------
    coef : torch.Tensor
        Model coefficients.
    std_errors : torch.Tensor
        Standard errors.
    alpha : float, default=0.05
        Significance level.
    dof : int, optional
        Degrees of freedom.
    distribution : str, default='t'
        Distribution type ('t' or 'z').

    Returns
    -------
    lower : np.ndarray
        Lower bounds.
    upper : np.ndarray
        Upper bounds.
    """
    coef_np = coef.detach().cpu().numpy()
    std_errors_np = std_errors.detach().cpu().numpy()

    if distribution == "t" and dof is not None:
        critical_value = stats.t.ppf(1 - alpha / 2, df=dof)
    elif distribution == "z":
        critical_value = stats.norm.ppf(1 - alpha / 2)
    else:
        # Default to normal distribution
        critical_value = stats.norm.ppf(1 - alpha / 2)

    margin_of_error = critical_value * std_errors_np
    lower = coef_np - margin_of_error
    upper = coef_np + margin_of_error

    return lower, upper


def compute_model_statistics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    log_likelihood: float,
    n_params: int,
    model_type: str = "linear",
) -> dict[str, float]:
    """
    Compute model fit statistics.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    log_likelihood : float
        Log-likelihood value.
    n_params : int
        Number of parameters.
    model_type : str, default='linear'
        Model type ('linear' or 'logistic').

    Returns
    -------
    stats_dict : dict
        Dictionary of model statistics.
    """
    n_obs = len(y_true)

    stats_dict = {
        "log_likelihood": log_likelihood,
        "aic": 2 * n_params - 2 * log_likelihood,
        "bic": n_params * np.log(n_obs) - 2 * log_likelihood,
        "n_obs": n_obs,
        "n_params": n_params,
    }

    if model_type == "linear":
        # R-squared and adjusted R-squared
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params)

        stats_dict.update(
            {
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "mse": ss_res / n_obs,
                "rmse": np.sqrt(ss_res / n_obs),
            }
        )

    elif model_type == "logistic":
        # Pseudo R-squared (McFadden's)
        # Assuming null log-likelihood is available
        # For now, compute a simple classification accuracy
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_binary = (y_pred > 0.5).float().detach().cpu().numpy()

        accuracy = np.mean(y_true_np == y_pred_binary)
        stats_dict["accuracy"] = accuracy

    return stats_dict


def format_summary_table(
    coef: torch.Tensor,
    std_err: torch.Tensor,
    feature_names: list,
    model_stats: dict[str, float],
    dof: int,
    distribution: str = "t",
) -> str:
    """
    Format a statistical summary table.

    Parameters
    ----------
    coef : torch.Tensor
        Model coefficients.
    std_err : torch.Tensor
        Standard errors.
    feature_names : list
        Feature names.
    model_stats : dict
        Model statistics.
    dof : int
        Degrees of freedom.
    distribution : str, default='t'
        Distribution type.

    Returns
    -------
    summary : str
        Formatted summary string.
    """
    # Compute test statistics and p-values
    test_stats = compute_test_statistics(coef, std_err, distribution)
    p_values = compute_p_values(test_stats, dof, distribution)

    # Compute confidence intervals
    lower, upper = compute_confidence_intervals(
        coef, std_err, dof=dof, distribution=distribution
    )

    # Convert to numpy for formatting
    coef_np = coef.detach().cpu().numpy()
    std_err_np = std_err.detach().cpu().numpy()
    test_stats_np = test_stats.detach().cpu().numpy()

    # Create summary table
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("                          Statistical Summary")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Coefficients table
    header = f"{'Variable':<15} {'coef':<10} {'std err':<10} {'t':<8} {'P>|t|':<8} {'[0.025':<10} {'0.975]':<10}"
    summary_lines.append(header)
    summary_lines.append("-" * 80)

    for i, name in enumerate(feature_names):
        line = (
            f"{name:<15} {coef_np[i]:>9.3f} {std_err_np[i]:>9.3f} "
            f"{test_stats_np[i]:>7.2f} {p_values[i]:>7.3f} "
            f"{lower[i]:>9.3f} {upper[i]:>9.3f}"
        )
        summary_lines.append(line)

    summary_lines.append("=" * 80)
    summary_lines.append("")

    # Model statistics
    if "r_squared" in model_stats:
        summary_lines.append(f"R-squared:         {model_stats['r_squared']:>8.3f}")
        summary_lines.append(f"Adj. R-squared:    {model_stats['adj_r_squared']:>8.3f}")
        if "mse" in model_stats:
            summary_lines.append(f"Mean Squared Error: {model_stats['mse']:>7.3f}")

    if "accuracy" in model_stats:
        summary_lines.append(f"Accuracy:          {model_stats['accuracy']:>8.3f}")

    summary_lines.append(f"Log-Likelihood:    {model_stats['log_likelihood']:>8.2f}")
    summary_lines.append(f"AIC:               {model_stats['aic']:>8.2f}")
    summary_lines.append(f"BIC:               {model_stats['bic']:>8.2f}")
    summary_lines.append(f"No. Observations:  {model_stats['n_obs']:>8}")
    summary_lines.append(f"Df Residuals:      {dof:>8}")
    summary_lines.append("")
    summary_lines.append("=" * 80)

    return "\n".join(summary_lines)


def compute_residuals(
    y_true: torch.Tensor, y_pred: torch.Tensor, residual_type: str = "raw"
) -> torch.Tensor:
    """
    Compute different types of residuals.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    residual_type : str, default='raw'
        Type of residuals ('raw', 'standardized', 'studentized').

    Returns
    -------
    residuals : torch.Tensor
        Computed residuals.
    """
    raw_residuals = y_true - y_pred

    if residual_type == "raw":
        return raw_residuals
    elif residual_type == "standardized":
        # Standardized residuals = residuals / sqrt(MSE)
        mse = torch.mean(raw_residuals**2)
        return raw_residuals / torch.sqrt(mse)
    elif residual_type == "studentized":
        # Simplified studentized residuals (without leverage)
        # For full studentized residuals, leverage values are needed
        mse = torch.mean(raw_residuals**2)
        return raw_residuals / torch.sqrt(mse)
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")


def compute_information_criteria(
    log_likelihood: float, n_params: int, n_obs: int
) -> dict[str, float]:
    """
    Compute various information criteria.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood value.
    n_params : int
        Number of parameters.
    n_obs : int
        Number of observations.

    Returns
    -------
    criteria : dict
        Dictionary of information criteria.
    """
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood

    # Corrected AIC for small samples
    if n_obs / n_params < 40:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
    else:
        aicc = aic

    return {"aic": aic, "bic": bic, "aicc": aicc}
