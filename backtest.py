import numpy as np

class WalkForwardBacktest:
    def __init__(self, window_size=252, step_size=63):
        self.window_size = window_size
        self.step_size = step_size

    def run(self, model_class, model_params, returns, transition_variable=None):
        T = len(returns)
        predictions = []
        actuals = []
        for start in range(0, T - self.window_size - self.step_size, self.step_size):
            end = start + self.window_size
            train_returns = returns[start:end]
            test_returns = returns[end:end+self.step_size]
            if transition_variable is not None:
                # For external transition, we need to pass it – simplified: use self-exciting
                pass
            model = model_class(**model_params)
            model.fit(train_returns)
            for i in range(min(len(test_returns), len(returns) - end)):
                hist = returns[:end + i]
                pred = model.predict(hist)
                predictions.append(pred)
                actuals.append(returns[end + i])
        return np.array(predictions), np.array(actuals)

    def evaluate(self, predictions, actuals):
        signals = np.sign(predictions)
        strategy_returns = signals * actuals
        metrics = {
            "mean_return": np.mean(strategy_returns) * 252,
            "volatility": np.std(strategy_returns) * np.sqrt(252),
            "sharpe": np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252),
            "hit_rate": np.mean(strategy_returns > 0),
            "max_drawdown": self._max_drawdown(strategy_returns)
        }
        return metrics

    def _max_drawdown(self, returns):
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
