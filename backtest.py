"""backtest.py — Walk-forward backtesting for STAR models."""

from __future__ import annotations

import numpy as np

from estar import ESTAR
from lstar import LSTAR
from setar import SETAR

MODEL_CLASSES = {"setar": SETAR, "lstar": LSTAR, "estar": ESTAR}


class WalkForwardBacktest:
    def __init__(self, window_size: int = 252, step_size: int = 63) -> None:
        self.window_size = window_size
        self.step_size = step_size

    def run(
        self,
        model_type: str,
        p: int,
        d: int,
        params: dict,
        returns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Walk-forward backtest — refits model on each training window."""
        cls = MODEL_CLASSES[model_type]
        y = returns
        T = len(y)
        predictions: list[float] = []
        actuals: list[float] = []

        for start in range(0, T - self.window_size - self.step_size, self.step_size):
            end = start + self.window_size
            train = y[start:end]

            try:
                model = cls(p=p, d=d)
                model.fit(train)
            except Exception:
                continue

            for i in range(self.step_size):
                idx = end + i
                if idx >= T:
                    break
                hist = y[:idx]
                try:
                    pred = float(np.squeeze(model.predict(hist)))
                except Exception:
                    pred = 0.0
                predictions.append(pred)
                actuals.append(float(y[idx]))

        return np.array(predictions), np.array(actuals)

    def evaluate(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> dict[str, float]:
        signals = np.sign(predictions)
        strategy_returns = signals * actuals
        std = np.std(strategy_returns)
        return {
            "mean_return": float(np.mean(strategy_returns) * 252),
            "volatility": float(std * np.sqrt(252)),
            "sharpe": float(np.mean(strategy_returns) / (std + 1e-8) * np.sqrt(252)),
            "hit_rate": float(np.mean(strategy_returns > 0)),
            "max_drawdown": float(self._max_drawdown(strategy_returns)),
        }

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + np.clip(returns, -0.5, 0.5))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (peak + 1e-8)
        return float(np.min(drawdown))
