# 🤖 Agentic Analytics Engine

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-1F77B4?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-9ACD32?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-FFD700?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</p>

---

# 🚀 Overview

**Agentic Analytics Engine** is a modular AI-powered analytics platform that enables users to analyze datasets using natural language.

Instead of writing code for statistics, visualizations, or machine learning, users simply ask questions such as:

> **"Predict customer churn"**

> **"Show histogram of Age"**

> **"What is the correlation between Age and Balance?"**

The system automatically understands the user's intent, routes the request to the appropriate analytics agent, performs the analysis, and generates professional insights.

---

# ✨ Current Features

## 🤖 AI Analyst

- Natural language analytics
- Conversational interface
- Intelligent request routing
- Downloadable analysis
- Agent execution trace

---

## 🧠 Semantic Analyzer

Automatically detects

- Identifier columns
- Numerical features
- Categorical features
- Target columns
- Feature columns

Supports intelligent target detection for datasets such as

- Customer Churn
- HR Attrition
- Loan Default
- Fraud Detection
- Titanic Survival

---

## 📊 Statistics Agent

Supports

- Descriptive Statistics
- Correlation Analysis
- Summary Statistics
- Statistical Planning
- Insight Generation
- Business Recommendations

---

## 📈 Visualization Agent

Automatically generates

- Histogram
- Scatter Plot
- Bar Chart
- Box Plot

along with AI-generated interpretations.

---

## 🤖 Machine Learning Agent

Automatically performs

- ML Problem Detection
- Classification / Regression Planning
- Automatic Target Detection
- Feature Selection
- Missing Value Handling
- One-Hot Encoding
- Feature Scaling
- Train/Test Split
- Multi-Model Training
- Best Model Selection
- Performance Evaluation
- AI-generated ML Insights

---

# 🏆 Supported Machine Learning Models

## Classification

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

---

## Regression

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
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 Statistics Agent   Visualization Agent   Machine Learning Agent
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                Insight Generation Layer
                           ▼
                  Streamlit AI Analyst
```

---

# 📂 Project Structure

```
Agentic-Analytics-Engine/

├── data/
│
├── src/
│   ├── agents/
│   │
│   ├── router.py
│   ├── router_graph.py
│   ├── semantic_analyzer.py
│   │
│   ├── statistical_planner.py
│   ├── statistical_executor.py
│   ├── statistical_graph.py
│   ├── statistical_insight.py
│   │
│   ├── visualization_planner.py
│   ├── visualization_executor.py
│   ├── visualization_graph.py
│   ├── visualization_insight.py
│   │
│   ├── ml_planner.py
│   ├── ml_executor.py
│   ├── ml_graph.py
│   └── ml_insight.py
│
├── src/tools/
│   ├── statistics.py
│   ├── visualization.py
│   └── ml_models.py
│
├── ui/
│   ├── streamlit_app.py
│   └── components/
│       ├── analyst.py
│       ├── analyst_chat.py
│       ├── analyst_router.py
│       ├── analyst_renderer.py
│       ├── analyst_statistics.py
│       ├── analyst_visualization.py
│       ├── analyst_ml.py
│       ├── analyst_download.py
│       ├── analyst_trace.py
│       └── analyst_state.py
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

# 💬 Example Questions

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

# 📊 Machine Learning Workflow

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
   │
   ▼
AI-generated Insights
```

---

# 🧪 Testing

The project includes dedicated tests for

- Semantic Analyzer
- Router
- Statistics Agent
- Visualization Agent
- ML Planner
- ML Executor
- ML Graph
- ML Insight
- Model Registry

Run all tests

```bash
pytest
```

---

# 🛠️ Tech Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Visualization

- Matplotlib
- Plotly

### Frontend

- Streamlit

---

# 🚀 Current Status (v0.3.0)

| Component | Status |
|-----------|--------|
| Semantic Analyzer | ✅ |
| Intelligent Router | ✅ |
| Statistics Agent | ✅ |
| Visualization Agent | ✅ |
| Machine Learning Agent | ✅ |
| AI Analyst UI | ✅ |
| Multi-Model Training | ✅ |
| Unit Tests | ✅ |

---

# 🗺️ Roadmap

### ✅ Completed

- Semantic Analyzer
- Intelligent Router
- Statistics Agent
- Visualization Agent
- Machine Learning Agent
- Streamlit AI Analyst
- Model Registry
- Natural Language Query Processing

### 🔄 Next Milestones

- SHAP Explainability
- Model Comparison Dashboard
- MLflow Experiment Tracking
- PDF Report Generation
- FastAPI Backend
- Docker Support
- GitHub Actions CI/CD

---

# 👨‍💻 Author

**Pratik Lagishetty**

GitHub: https://github.com/Blvackk

---

⭐ If you found this project useful, consider giving it a star!
