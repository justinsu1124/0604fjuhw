"""RSRS (Resistance-Support Relative Strength) custom indicator for NautilusTrader.

Rolling OLS:  High = α + β · Low + ε   (window = ols_window)
Z-score:      beta_z = (β − μ_β) / σ_β  (window = zscore_window)
Signal:       beta_z × R² × β            (right-skewed)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from nautilus_trader.indicators import Indicator
from nautilus_trader.model.data import Bar


class RSRSIndicator(Indicator):
    """RSRS right-skewed timing indicator."""

    def __init__(
        self,
        ols_window: int = 18,
        zscore_window: int = 1100,
    ) -> None:
        super().__init__([])
        self.ols_window = ols_window
        self.zscore_window = zscore_window

        self._highs: deque[float] = deque(maxlen=ols_window)
        self._lows: deque[float] = deque(maxlen=ols_window)
        self._betas: deque[float] = deque(maxlen=zscore_window)
        self._count = 0

        self.beta: float = 0.0
        self.rsquare: float = 0.0
        self.tvalue: float = 0.0
        self.beta_z: float = 0.0
        self.signal_value: float = 0.0
        self.signal_standard: float = 0.0
        self.signal_r2adj: float = 0.0
        self.signal_tvalue_adj: float = 0.0
        self.signal_tvalue_right: float = 0.0
        self._tvalue_sum: float = 0.0
        self._tvalue_count: int = 0

    @property
    def warmup_period(self) -> int:
        return self.ols_window + self.zscore_window

    def handle_bar(self, bar: Bar) -> None:  # noqa: D401
        self.update_raw(float(bar.high), float(bar.low))

    def update_raw(self, high: float, low: float) -> None:
        self._highs.append(high)
        self._lows.append(low)
        self._count += 1

        if not self.has_inputs:
            self._set_has_inputs(True)

        if self._count < self.ols_window:
            return

        h = np.array(self._highs)
        low_arr = np.array(self._lows)
        ones = np.ones(len(low_arr))
        X = np.column_stack([ones, low_arr])

        result = np.linalg.lstsq(X, h, rcond=None)
        coeffs = result[0]
        self.beta = coeffs[1]

        y_pred = X @ coeffs
        ss_res = np.sum((h - y_pred) ** 2)
        ss_tot = np.sum((h - np.mean(h)) ** 2)
        self.rsquare = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # t-statistic of beta
        n = len(h)
        mse = ss_res / max(n - 2, 1)
        x_mean = low_arr.mean()
        sx2 = np.sum((low_arr - x_mean) ** 2)
        se_beta = np.sqrt(mse / sx2) if sx2 > 1e-12 else 1e-12
        self.tvalue = self.beta / se_beta if se_beta > 1e-12 else 0.0
        self._tvalue_sum += self.tvalue
        self._tvalue_count += 1

        self._betas.append(self.beta)

        if len(self._betas) >= self.zscore_window:
            betas_arr = np.array(self._betas)
            mean_b = betas_arr.mean()
            std_b = betas_arr.std(ddof=1)
            self.beta_z = (self.beta - mean_b) / std_b if std_b > 1e-12 else 0.0
            self.signal_standard = self.beta_z
            self.signal_r2adj = self.beta_z * self.rsquare
            self.signal_value = self.beta_z * self.rsquare * self.beta
            t_mean = self._tvalue_sum / max(self._tvalue_count, 1)
            self.signal_tvalue_adj = (self.tvalue * self.beta_z) / t_mean if t_mean > 1e-12 else 0.0
            self.signal_tvalue_right = (self.tvalue * self.beta_z * self.beta) / t_mean if t_mean > 1e-12 else 0.0

            if not self.initialized:
                self._set_initialized(True)

    def _reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._betas.clear()
        self._count = 0
        self.beta = 0.0
        self.rsquare = 0.0
        self.tvalue = 0.0
        self.beta_z = 0.0
        self.signal_value = 0.0
        self.signal_standard = 0.0
        self.signal_r2adj = 0.0
        self.signal_tvalue_adj = 0.0
        self.signal_tvalue_right = 0.0
        self._tvalue_sum = 0.0
        self._tvalue_count = 0
