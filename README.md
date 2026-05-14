# Stock Market Machine Learning Algorithm 📈🤖
### Temporal Diffusion Model vs LSTM + Transformer on Large-Scale Financial Data

A comparative deep learning study exploring whether a **temporal diffusion model pre-weighted by a fine-tuned BERT encoder** can match or outperform a traditional LSTM + Transformer architecture for stock market prediction. Trained on 900k+ samples across historical price and sentiment data using NVIDIA A100 GPUs.

> **Key Result:** Achieved a **6% improvement over the LSTM + Transformer baseline** by combining diffusion-based temporal modeling with NLP-derived sentiment features.

---

## 🔍 Overview

Stock market prediction is challenging due to high noise, non-stationarity, and the influence of external factors like news sentiment. This project tackles that by:

1. **Fine-tuning a BERT encoder** on financial news to extract sentiment features
2. **Pre-weighting a temporal diffusion model** with those sentiment embeddings
3. **Comparing against an LSTM + Transformer baseline** across multiple market conditions

---

## 🏗️ Architecture

```
Financial News + Headlines
        │
        ▼
Fine-tuned BERT Encoder
  └── Sentiment embeddings
        │
Historical Price Data        │
+ Volatility Indicators ─────┤
        │                    │
        ▼                    ▼
    ┌─────────────────────────────┐
    │  Temporal Diffusion Model   │
    │  (pre-weighted by BERT)     │
    └─────────────────────────────┘
              │
              ▼
       Price Prediction
```

---

## 📊 Results

| Model | Performance vs Baseline |
|---|---|
| LSTM + Transformer (baseline) | — |
| Temporal Diffusion + BERT (ours) | +6% improvement |

Evaluated across varying market conditions and feature configurations to test generalization.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Language Model | BERT (fine-tuned) |
| Architecture | Temporal Diffusion Model, LSTM + Transformer |
| Hardware | NVIDIA A100 GPU |
| Dataset | 900k+ samples — historical prices, volatility indicators, NLP sentiment |
| Course | ECE534 — Distributed Deep Learning, Rutgers University |

---

## 🗃️ Data Pipeline

- **Historical prices** — OHLCV data across multiple tickers
- **Volatility indicators** — computed from price history
- **Sentiment features** — NLP-derived from financial news headlines via fine-tuned BERT
- Combined into a unified feature matrix for model input

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- NVIDIA GPU recommended (A100 used for training)

### Installation

```bash
git clone https://github.com/vir222/stock-market-ml.git
cd stock-market-ml
pip install -r requirements.txt
```

### Training

```bash
# Train the diffusion model
python train.py --model diffusion --epochs 50

# Train the baseline
python train.py --model lstm_transformer --epochs 50

# Evaluate and compare
python evaluate.py
```

---

## 👥 Contributors

| Name | Email |
|---|---|
| Vir Vaidya | vv275@rutgers.edu |
| Abdul Wadud Abbasi | wa176@rutgers.edu |
| Aman Patel | amp181@rutgers.edu |
| Sohom Pal | sp2160@rutgers.edu |

---

## 👤 Contact

**Vir Vaidya** — vv275@rutgers.edu  
M.S. Computer Engineering, Rutgers University  
Course: ECE534 Distributed Deep Learning
