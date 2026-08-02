# 🤖 Agentic Analytics Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg">
  <img src="https://img.shields.io/badge/Streamlit-WebApp-red.svg">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg">
  <img src="https://img.shields.io/badge/XGBoost-Enabled-green.svg">
  <img src="https://img.shields.io/badge/LightGBM-Enabled-brightgreen.svg">
  <img src="https://img.shields.io/badge/Status-Active-success.svg">
</p>

## 🚀 Overview

**Agentic Analytics Engine** is an AI-powered analytics platform that allows users to analyze datasets using natural language.

Instead of manually writing code for EDA, statistics, visualizations, or machine learning, users simply ask questions in plain English.

The system automatically understands the intent, routes the request to the appropriate AI agent, executes the analysis, and returns professional insights.

---

# ✨ Features

### 🧠 AI Analyst

- Natural Language Analytics
- Chat-based interface
- Intelligent query routing

Example:

> Predict customer churn

> Show histogram of Age

> What is the correlation between Age and Balance?

---

### 🧩 Semantic Analyzer

Automatically detects

- Identifier columns
- Numerical features
- Categorical features
- Target columns
- Feature columns

Supports intelligent target detection for datasets like:

- Customer Churn
- HR Attrition
- Titanic Survival
- Loan Default
- Fraud Detection

---

### 📊 Statistics Agent

Automatically performs

- Descriptive Statistics
- Correlation Analysis
- Statistical Insights
- Business Recommendations

---

### 📈 Visualization Agent

Automatically generates

- Histograms
- Scatter Plots
- Bar Charts
- Box Plots

with AI-generated interpretations.

---

### 🤖 Machine Learning Agent

Supports automatic

- ML Problem Detection
- Target Detection
- Feature Selection
- Data Preprocessing
- Missing Value Handling
- Feature Scaling
- One-Hot Encoding
- Train/Test Split
- Model Training
- Best Model Selection

---

## 🏆 Supported Models

### Classification

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

---

# 🏗️ Architecture

```
                  User Question
                         │
                         ▼
               Semantic Analyzer
                         │
                         ▼
                 Intelligent Router
         ┌───────────┼───────────┐
         ▼           ▼           ▼
 Statistics Agent Visualization ML Agent
         │           │           │
         └───────────┼───────────┘
                     ▼
              AI Generated Insights
                     ▼
              Streamlit Dashboard
```

---

# 📂 Project Structure

```
Agentic-Analytics-Engine/

├── data/
├── src/
│
├── agents/
│   ├── router.py
│   ├── router_graph.py
│   ├── semantic_analyzer.py
│   ├── statistical_*.py
│   ├── visualization_*.py
│   ├── ml_planner.py
│   ├── ml_executor.py
│   ├── ml_graph.py
│   └── ml_insight.py
│
├── tools/
│   ├── statistics.py
│   ├── visualization.py
│   └── ml_models.py
│
├── ui/
│   ├── streamlit_app.py
│   └── components/
│
├── tests/
│
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/Blvackk/Agentic-Analytics-Engine.git

cd Agentic-Analytics-Engine

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# ▶️ Run

```bash
streamlit run ui/streamlit_app.py
```

---

# 🧪 Example Questions

## Statistics

```
What is the correlation between Age and Balance?
```

```
Show summary statistics for CreditScore
```

---

## Visualization

```
Show histogram of Age
```

```
Plot Age vs Balance
```

---

## Machine Learning

```
Predict customer churn
```

```
Build a churn prediction model
```

```
Train a classification model
```

---

# 📊 Machine Learning Pipeline

```
Dataset
    │
    ▼
Semantic Analysis
    │
    ▼
Target Detection
    │
    ▼
Feature Selection
    │
    ▼
Preprocessing
    │
    ▼
Train/Test Split
    │
    ▼
Multiple Models
    │
    ▼
Performance Evaluation
    │
    ▼
Best Model Selection
```

---

# 🧪 Testing

The project includes unit tests for

- Semantic Analyzer
- Router
- Statistics Agent
- Visualization Agent
- Machine Learning Planner
- Machine Learning Executor
- Machine Learning Graph
- Model Registry

Run all tests

```bash
pytest
```

---

# 🚀 Roadmap

### ✅ Completed

- AI Analyst
- Semantic Analyzer
- Statistics Agent
- Visualization Agent
- Machine Learning Agent
- Model Registry
- Streamlit UI

### 🔄 In Progress

- Model Comparison Dashboard
- SHAP Explainability
- MLflow Experiment Tracking
- PDF Report Generation

### 📅 Planned

- FastAPI Backend
- Docker Support
- GitHub Actions CI/CD
- Multi-Agent Orchestration
- LLM-powered Recommendations

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Science

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

### Visualization

- Matplotlib
- Plotly

### Frontend

- Streamlit

### AI

- Rule-Based Router
- Semantic Analysis
- Multi-Agent Architecture

---

# 📜 License

MIT License

---

# 👨‍💻 Author

**Pratik Lagishetty**

GitHub

https://github.com/Blvackk

---

⭐ If you found this project useful, consider giving it a star!
