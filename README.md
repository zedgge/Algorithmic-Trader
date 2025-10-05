# Algorithmic Trading and Machine Learning System

**Author:** [Your Name]  
**Technologies:** Python, PyTorch (CUDA), Pandas, NumPy, XGBoost, Joblib, Alpaca API, Yahoo Finance, Vector Databases, Bloomberg API (integration planned)

---

## Overview

This repository documents the development of a **standalone trading and research application** I built from scratch.  
It was designed to perform **automated trading, predictive modeling, and quantitative analysis** across multiple markets — equities, crypto, and forex — using both rule-based logic and machine learning models.

> **Note:** The original production code and datasets are not included.  
> A hardware failure led to the loss of critical files, so this repository serves as a technical record, showing snippets, logic outlines, and architectural design.  
> The intent is to document the system’s structure and methods, not to release an active trading platform.

---

## Motivation and Context

This system was built independently while I was taking advanced coursework at **MIT** and **Harvard** to strengthen my quantitative, data science, and systems engineering background.  
The goal was to design a framework that could **analyze live data, predict directional movements, and execute trades** with proper **risk management** and **capital preservation logic** — all without manual input.

The project was structured around real-world trading constraints:
- Market hours, liquidity windows, and time zone alignment
- Order routing delays and API latency
- Capital allocation across correlated and non-correlated instruments
- Automated risk adjustment based on volatility and drawdown limits

---

## System Structure

The application was divided into four major layers:

### 1. Data Layer
- Pulled live and historical data from **Yahoo Finance** and **Alpaca**, with planned support for **Bloomberg API**.  
- Handled real-time stream updates, data normalization, and storage into **vector databases** for quick historical pattern matching.  
- Engineered technical features: moving averages, volume-weighted price action, volatility bands, and momentum ratios.  

### 2. Modeling Layer
- Used **Temporal Fusion Transformers (TFT)** for time-series prediction and **XGBoost** for trend classification.  
- Incorporated a hybrid model ensemble to cross-verify predicted movements.  
- Models were trained and serialized using **Joblib** for fast loading during live sessions.  
- Implemented a **confidence weighting system**: when predictions conflicted, execution depended on model reliability and market volatility.

### 3. Execution Layer
- Used **Alpaca’s API** for trade execution.  
- Supported simultaneous operation across multiple markets.  
- Contained pre-trade checks (available margin, exposure limits) and post-trade validation (confirmation, slippage logging).  
- Execution routing was designed to minimize latency and API rate-limit issues.  
- Included configurable modes:
  - **Automated trading** (fully autonomous)
  - **Assisted trading** (user confirmation before execution)
  - **Research mode** (no live trades, simulation only)

### 4. Risk and Capital Management
- Implemented adjustable **risk ratios** based on account equity.  
- Used rolling volatility measures (ATR and variance tracking) to scale position sizes.  
- Stop-loss and trailing stop parameters were dynamically calculated.  
- Built a fail-safe system to stop trading after a defined drawdown or connectivity issue.  
- Used time-based shutdown logic before major news events or at session end to avoid after-hours volatility.

---

## Machine Learning and Quant Methods

- **Framework:** PyTorch with CUDA acceleration for model training and inference.  
- **Data:** Historical OHLCV data combined with technical indicators and calendar-based features (market open/close, weekdays, earnings days).  
- **Models:**  
  - **TFT (Temporal Fusion Transformer)** for long-range dependencies in price movement  
  - **XGBoost** for short-term directional prediction  
  - Ensemble averaging for stability across instruments  
- **Evaluation:**  
  - Custom validation metric combining accuracy, profit factor, and drawdown ratio  
  - Backtesting framework to simulate trades before deployment  
- **Vector Search:** Used embedding comparisons for historical similarity queries (“find past conditions that match the current state”)  

---

## Core Features

- **Multi-Market Operation:** Supports equities, forex, and crypto with separate strategy pipelines.  
- **Time-Aware Trading:** System pauses or closes trades around market open/close and known high-volatility windows.  
- **Adaptive Algorithms:** System dynamically switches between models or disables trading when uncertainty exceeds thresholds.  
- **Capital Efficiency:** Positions are sized to preserve capital and maintain balanced exposure.  
- **Custom Risk Profiles:** Risk and position parameters could be adjusted per asset or per market.  
- **Event Logging:** All trade decisions, data fetches, and model inferences logged to local storage for review.

---

## Example Components

> *(Snippets and screenshots can be added here.)*  
> Suggested content:
> - Example signal-generation function  
> - Model ensemble decision logic  
> - GUI elements (portfolio view, trade monitor, alerts)  
> - Example trade summary output  

---

## Design Principles

1. **Transparency:** Every trade decision could be traced back to data and model reasoning.  
2. **Safety First:** All automation functions had risk caps and override switches.  
3. **Scalability:** Designed to handle multiple accounts and markets with minimal performance loss.  
4. **Low Latency:** Execution functions were written to reduce overhead and response times.  
5. **Extensibility:** Modular design allows new data feeds or algorithms to be plugged in easily.

---

## Why the Repository Exists

This project reflects my personal approach to quantitative research, systems design, and applied machine learning.  
Although the original files were partially lost, the concepts, structure, and technical methods documented here show:
- Real understanding of trading system architecture  
- Experience with financial data engineering and predictive modeling  
- Awareness of how ML interacts with real market structure, latency, and risk management  
- Ability to design and maintain complex, reliable software independently

---

## Next Steps

I plan to rebuild the system using more efficient architectures:
- Transformer-based hybrid models with online learning capability  
- Real-time inference pipelines using GPU batching  
- Expanded market data coverage and automatic regime detection  

The long-term goal is to evolve this project into a **modular trading and research framework** that could adapt to different strategies and data sources in real time.

---

## Contact

**Email:** [your_email@domain.com]  
**LinkedIn:** [Your LinkedIn Profile]  
**GitHub:** [Your GitHub Profile]  

---

> *“Markets are not random — they are structured chaos.  
> The goal isn’t to predict perfectly, but to measure uncertainty better than others.”*
