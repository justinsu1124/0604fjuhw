# RSRS 阻力支撐相對強度擇時策略 — NautilusTrader 復現

> 復現光大證券《基於阻力支撐相對強度(RSRS)的市場擇時》研報，  
> 以 **NautilusTrader** 框架實作回測，數據來源 **FMP**，  
> 並加入 **成交量 / MA 確認** 濾網。

## 策略核心

| 步驟 | 公式 | 說明 |
|------|------|------|
| Rolling OLS (N=18) | `High = α + β·Low + ε` | β = RSRS 斜率，R² = 擬合優度 |
| Z-score (M=1100) | `beta_z = (β − μ_β) / σ_β` | 標準化比較歷史相對強弱 |
| Right-skew signal | `signal = beta_z × R² × β` | 右偏修正擇時信號 |
| Volume confirmation | `volume > SMA(volume, 20)` | 確認量能配合 |

**交易規則**（長倉擇時，不做空）：

- **買入**：`signal > 0.7` **且** 當日成交量 > 20日均量
- **賣出**：`signal < −0.7`
- 其他情況維持原有部位

## 專案結構

```
├── backtest.py          # 主程式：載入數據 → 建引擎 → 跑回測 → 輸出報告
├── rsrs_indicator.py    # 自定義 RSRS Indicator（OLS + Z-score + R²）
├── rsrs_strategy.py     # NautilusTrader Strategy（信號 + 量能確認 + 下單）
├── data_loader.py       # FMP 數據載入 → NautilusTrader Bar 轉換
├── requirements.txt     # 依賴套件
└── results/             # 回測結果（equity curve、fills、positions）
```

## 快速開始

```bash
pip install -r requirements.txt

# 設定 FMP API Key
export FMP_API_KEY="your_key_here"

# 執行回測（預設 SPY 2000-01-01 至今）
python backtest.py

# 自定義參數
python backtest.py --start 2005-01-01 --ols-window 18 --zscore-window 1100 \
    --buy-threshold 0.7 --sell-threshold -0.7 --vol-ma-period 20
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--start` | 2000-01-01 | 數據起始日期 |
| `--end` | 最新 | 數據結束日期 |
| `--ols-window` | 18 | OLS 回歸窗口 N |
| `--zscore-window` | 1100 | Z-score 標準化窗口 M |
| `--buy-threshold` | 0.7 | 買入閾值 |
| `--sell-threshold` | -0.7 | 賣出閾值 |
| `--vol-ma-period` | 20 | 成交量均線週期 |
| `--trade-size` | 100 | 每次交易股數 |

## 可延伸方向

- 進出場門檻不對稱（如買 0.7、賣 -0.5）
- 波動率過濾（ATR / VIX 過高時降倉）
- 多標的輪動（SPY + QQQ + IWM）
- t-value 修正信號變體
- Walk-forward 最佳化 N, M 參數

## 參考

- 光大證券《基於阻力支撐相對強度(RSRS)的市場擇時》(2017)
- [NautilusTrader Documentation](https://nautilustrader.io/)
- [FMP API](https://financialmodelingprep.com/)
