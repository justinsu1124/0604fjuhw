"""RSRS timing strategy for NautilusTrader with volume/MA confirmation.

Logic
-----
- Compute right-skewed RSRS signal each bar.
- Buy when  signal > buy_threshold  **and** volume > SMA(volume, vol_ma_period).
- Sell when signal < sell_threshold.
- Hold previous position otherwise.
- Long only, no short.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from rsrs_indicator import RSRSIndicator


class RSRSStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the RSRS timing strategy."""

    bar_type_str: str = "SPY.XNAS-1-DAY-LAST-EXTERNAL"
    instrument_id_str: str = "SPY.XNAS"
    ols_window: int = 18
    zscore_window: int = 1100
    buy_threshold: float = 0.7
    sell_threshold: float = -0.7
    trade_size: int = 100
    vol_ma_period: int = 20


class RSRSStrategy(Strategy):
    """RSRS right-skewed timing with volume/MA confirmation."""

    def __init__(self, config: RSRSStrategyConfig) -> None:
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type_str)
        self.instrument_id = InstrumentId.from_str(config.instrument_id_str)
        self.trade_size = config.trade_size

        self.buy_threshold = config.buy_threshold
        self.sell_threshold = config.sell_threshold

        self.rsrs = RSRSIndicator(
            ols_window=config.ols_window,
            zscore_window=config.zscore_window,
        )

        self.vol_ma_period = config.vol_ma_period
        self._volumes: deque[float] = deque(maxlen=config.vol_ma_period)

        self._is_long = False

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)
        self.log.info(
            f"RSRS strategy started | "
            f"OLS={self.rsrs.ols_window} Z={self.rsrs.zscore_window} "
            f"buy={self.buy_threshold} sell={self.sell_threshold} "
            f"vol_ma={self.vol_ma_period}",
        )

    def on_bar(self, bar: Bar) -> None:
        self.rsrs.handle_bar(bar)

        vol = float(bar.volume)
        self._volumes.append(vol)

        if not self.rsrs.initialized:
            return

        signal = self.rsrs.signal_value

        vol_ma = np.mean(self._volumes) if len(self._volumes) == self.vol_ma_period else None
        vol_confirmed = vol_ma is not None and vol > vol_ma

        if signal > self.buy_threshold and vol_confirmed and not self._is_long:
            self._enter_long()
        elif signal < self.sell_threshold and self._is_long:
            self._exit_long()

    def _enter_long(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=Quantity.from_int(self.trade_size),
        )
        self.submit_order(order)
        self._is_long = True
        self.log.info(
            f"BUY signal={self.rsrs.signal_value:.4f} "
            f"beta={self.rsrs.beta:.4f} R²={self.rsrs.rsquare:.4f}",
        )

    def _exit_long(self) -> None:
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=Quantity.from_int(self.trade_size),
        )
        self.submit_order(order)
        self._is_long = False
        self.log.info(
            f"SELL signal={self.rsrs.signal_value:.4f} "
            f"beta={self.rsrs.beta:.4f} R²={self.rsrs.rsquare:.4f}",
        )

    def on_stop(self) -> None:
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        self.rsrs.reset()
        self._volumes.clear()
        self._is_long = False
