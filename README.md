# Machine Learning Algorithmic Trading System

> **⚠️ Repository Status:** Currently undergoing reconstruction following hardware failure. Core functionality is being rebuilt with enhanced performance optimizations. Code samples and technical documentation reflect the improved architecture.

---

### Overview
This project is a fully custom-built algorithmic trading application designed and developed independently while completing advanced coursework through MITx and HarvardX.  
It combines real-time data ingestion, machine learning–based decision-making, and built-in risk management to operate autonomously across multiple markets.  

The system achieved a **12.2% ROI with a 56% win rate** over four months of live paper trading.  
It includes a complete framework for both **automated trading** and **data-driven decision support**, offering the flexibility to run as a self-managed trading agent or as a manual trading interface with integrated analytics.

---

### Core Features
- **Multi-Market Trading:** Supports simultaneous trading across multiple markets and asset classes.  
- **Machine Learning Integration:** Utilizes PyTorch, XGBoost, and TFT architectures for adaptive prediction and signal generation.  
- **Risk Management Engine:** Built-in configurable safeguards and position-sizing controls that adapt to market volatility.  
- **Asynchronous Continuous Learning:**  
  Outside active market sessions, the system entered a continuous-learning phase.  
  During this time, it ingested post-market data, recomputed rolling features, and fine-tuned model parameters against the most recent regime conditions.  
  This asynchronous training loop helped minimize performance decay caused by data drift and ensured that models were recalibrated before the next market open.  
  This process also included adaptive threshold updates for signal confidence and volatility-based position sizing recalibration.
- **Custom Paper Trader:** A built-in paper trading module used for model training, backtesting, and off-hour parameter tuning.  
- **News & Data Fusion:** Integrated financial news sentiment and historical data analysis to identify trade opportunities and generate alerts.  
- **Performance & Latency Optimization:** Designed for low-latency execution using CUDA acceleration and vectorized operations.  

---

### Technical Highlights
- **Languages & Frameworks:** Python, PyTorch, Pandas, NumPy, Joblib, XGBoost, and Alpaca API.  
- **Data Handling:** Efficient use of rolling datasets, historical caching, and on-the-fly normalization to optimize model input pipelines.  
- **APIs & Data Sources:** Integrated Alpaca and Yahoo Finance, with partial Bloomberg API integration in progress.  
- **GPU Acceleration:** Configured for multi-GPU use with CUDA optimization for training and inference workloads.  
- **Fail-Safes:** Included stop-loss, trade validation layers, and configurable capital exposure limits.  
- **System Architecture:** Modular design with isolated data, model, and execution pipelines for stability and scalability.  

For detailed implementation samples, see [TECHNICAL_HIGHLIGHTS.md](TECHNICAL_HIGHLIGHTS.md).

---

### Market Knowledge Integration
The system incorporates market session logic, volatility-based position scaling, and dynamic exposure control depending on liquidity and price action patterns.  
It factors in market open/close times, pre/post-market conditions, and macro-driven volatility shifts.  
Signal generation was influenced by both price momentum and volume correlation patterns, rather than relying on static thresholds.  

---

### Performance Metrics

**Live Paper Trading Results (4 months):**
- ROI: 12.2%
- Win Rate: 56%
- Multi-market execution across multiple asset classes

**System Performance:**
- Technical indicator computation: 50-100x speedup via Numba JIT
- Model training: 10-15x faster with mixed precision
- Inference latency: <5ms per batch

---

### Future Development
Planned improvements include:
- Completing Bloomberg API integration for higher-quality financial data access.  
- Enhanced GPU support for model training and inference optimization.  
- Performance balancing for multi-node or cluster deployments.  
- Expanded large-data handling for higher-frequency and multi-market scaling.  
- Microsecond-level interaction logging for both user actions and ML decisions.  
- Major performance optimizations for lower latency and higher throughput without hardware upgrades.  
- Redesigned, modular, and customizable GUI for a more intuitive user experience.  
- Expanded security layer for data protection, access control, and model integrity.  

---

### Repository Information
The full source code is not publicly available due to ongoing reconstruction and protection of proprietary logic developed during testing.  
Select snippets and architecture documentation are provided in [TECHNICAL_HIGHLIGHTS.md](TECHNICAL_HIGHLIGHTS.md) to demonstrate implementation depth, structure, and technical proficiency.

---

### Acknowledgments
Built independently as part of self-directed quantitative research and development while completing advanced coursework through MITx and HarvardX.  
This repository serves as a reference for employers and collaborators interested in quantitative systems, ML-based trading, and performance optimization.

---

### Contact
If you're interested in discussing quant development, ML modeling, or system architecture, feel free to reach out directly via LinkedIn or through GitHub.
