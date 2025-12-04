# 🌧️ Rainfall Prediction System  
**A Transformer-based Rainfall Forecasting System with Web Interface & LLM Weather Assistant**

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.9+-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.12+-orange)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

This project builds a rainfall prediction system using publicly available weather datasets (temperature, humidity, pressure, wind speed, rainfall, etc.).  
It includes **data preprocessing, feature correlation analysis, Transformer-based time-series prediction, professional normalization, a web UI**, and an **LLM-powered weather assistant**.

If a user asks irrelevant/non-weather questions, the AI will **politely refuse to answer**, keeping the system domain-specific.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Datasets & Processing](#-datasets--processing)
- [Model Description (Transformer)](#-model-description-transformer)
- [Environment Setup](#-environment-setup)
- [Model Training](#-model-training)
- [Prediction Examples](#-prediction-examples)
- [Web Interface & LLM Assistant](#-web-interface--llm-assistant)
- [Project Structure](#-project-structure)
- [Visualization Samples](#-visualization-samples)
- [TODO](#-todo)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 Project Overview

This project aims to provide an end-to-end rainfall forecasting platform using machine learning and modern deep learning techniques.

It includes:

- Public weather dataset ingestion  
- Data cleaning, transformation, and feature correlation analysis  
- Transformer-based time-series rainfall prediction  
- Web-based visualization interface  
- LLM module for natural language weather Q&A  
- Automatic refusal of non-weather-related questions  

---

## 🚀 Key Features

✔ Weather data cleaning & preprocessing  
✔ Correlation analysis between meteorological variables and rainfall  
✔ Transformer-based deep learning model  
✔ Professional normalization (Min-Max / Z-score) + inverse transformation  
✔ Daily rainfall forecasting  
✔ Web visualization interface  
✔ LLM weather assistant & question answering  
✔ Rejects unrelated questions automatically  

---

## 📊 Datasets & Processing

We use public meteorological datasets (e.g., NOAA, Australian Weather Dataset, National Meteorological Open Data), including:

- Temperature (max / min / mean)
- Relative humidity
- Air pressure
- Wind speed / wind direction
- Rainfall (target variable)
- Sunshine duration  
- Other auxiliary weather indicators

### 🔧 Data Processing Pipeline

- Handling missing values (interpolation, mean fill, etc.)
- Outlier detection and removal
- Time-series alignment and normalization
- Feature construction (window features, gradients, rolling statistics)
- Pearson correlation & mutual information analysis to identify rainfall-relevant factors

---

## 🧠 Model Description (Transformer)

We use a **Time-Series Transformer** as the main forecasting model.

### Model Highlights:

- Multi-Head Self-Attention  
- Captures long-term dependencies across meteorological variables  
- Supports multivariate inputs (temperature, humidity, pressure, etc.)  
- Superior to LSTM/GRU for complex nonlinear weather data  

### 📐 Normalization

We apply Min-Max or Z-score normalization:

- Ensures consistent feature scales  
- Stabilizes gradients and speeds up convergence  
- Inference results are inverse-transformed to obtain real rainfall values (mm)

---

## 🛠 Environment Setup

Recommended: **Python 3.9+**

### Install core dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
### For Web UI + LLM assistant
pip install gradio openai
# or
pip install streamlit
### Model Training
python train_daily_model.py
### Web Interface & LLM Assistant
streamlit run chat_app.py

## 🌐 Features

- **Natural language rainfall forecasting**
- **Weather knowledge Q&A**
- **Graphical visualization of predictions**
- **Automatic refusal of out-of-domain questions**

---

## 💬 Example Questions

- “Will it rain tomorrow?”  
- “How does humidity affect rainfall?”  
- “Which features are most correlated with precipitation?”  
- “Explain how the Transformer model works for weather forecasting.”  

