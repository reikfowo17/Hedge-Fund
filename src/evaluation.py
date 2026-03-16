import numpy as np


def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))


def weighted_rmse_score(y_target, y_pred, w) -> float:
    y_target = np.asarray(y_target, dtype=np.float64)
    y_pred   = np.asarray(y_pred, dtype=np.float64)
    w        = np.asarray(w, dtype=np.float64)

    denom = np.sum(w * y_target ** 2)
    if denom == 0:
        return 0.0

    ratio   = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val     = 1.0 - clipped

    return float(np.sqrt(val))


def evaluate_per_horizon(df, pred_col="prediction", target_col="y_target",
                         weight_col="weight", horizon_col="horizon"):
    scores = {}
    for h in sorted(df[horizon_col].unique()):
        mask = df[horizon_col] == h
        score = weighted_rmse_score(
            df.loc[mask, target_col],
            df.loc[mask, pred_col],
            df.loc[mask, weight_col],
        )
        scores[h] = score
        print(f"  Horizon {h:2d}: {score:.5f}")

    overall = weighted_rmse_score(
        df[target_col], df[pred_col], df[weight_col]
    )
    scores["overall"] = overall
    print(f"  {'Overall':>10s}: {overall:.5f}")

    return scores
