
# Agentic Analytics Engine

An AI-powered autonomous data analytics system that takes a raw CSV dataset and performs an end-to-end analytical workflow including data profiling, quality assessment, automated cleaning, semantic column understanding, statistical analysis, EDA planning, visualisation, AI-generated insights, and report generation.

The workflow is orchestrated using **LangGraph**, while a local **Ollama LLM** provides semantic reasoning and analytical interpretation.

---

## Overview

Traditional exploratory data analysis requires analysts to manually inspect datasets, identify data-quality problems, clean the data, select appropriate analyses, generate visualisations, interpret results, and prepare reports.

The **Agentic Analytics Engine** automates this workflow through specialised components coordinated as an analytics pipeline.

Instead of sending raw data directly to an LLM, the system uses Python-based analytical tools to calculate statistics and perform transformations first. The LLM then interprets structured analytical results to generate grounded insights.

---

## Phase 1 — Automated Analytics Pipeline

Phase 1 implements the complete automated analytics workflow:

```text
CSV Dataset
     │
     ▼
Dataset Loading
     │
     ▼
Dataset Profiling
     │
     ▼
Data Quality Analysis
     │
     ▼
Cleaning Required?
   /        \
 Yes         No
  │           │
  ▼           │
Cleaning      │
  │           │
  ▼           │
Validation    │
   \         /
     ▼
Semantic Analysis
     │
     ├── Identifier Detection
     ├── Target Detection
     └── Feature Detection
     │
     ▼
Statistical Analysis
     │
     ▼
EDA Planner
     │
     ▼
EDA Executor
     │
     ├── Histograms
     ├── Boxplots
     ├── Bar Charts
     ├── Scatter Plots
     └── Correlation Heatmap
     │
     ▼
Target Analysis
     │
     ▼
LLM Insight Generation
     │
     ▼
Report Generation
     │
     ▼
Streamlit Dashboard
```

---

## Key Features

### Automated Dataset Profiling

The engine automatically examines the structure of an uploaded dataset, including:

- Number of rows and columns
- Column names
- Data types
- Missing values
- Unique values
- Duplicate records

### Data Quality Detection

The quality-analysis component detects issues such as:

- Missing values
- Duplicate rows
- Potential structural problems
- Columns requiring cleaning

The workflow can conditionally decide whether the cleaning stage is required.

### Automated Data Cleaning

Supported data-quality problems can be cleaned automatically.

The system records:

- Original row count
- Cleaned row count
- Rows removed
- Missing values before and after cleaning
- Duplicate records before and after cleaning
- Cleaning operations performed

A validation stage verifies the result before downstream analysis.

### Semantic Column Analysis

Column data types alone do not describe their analytical meaning.

The semantic analyzer combines deterministic rules and LLM reasoning to classify columns into roles such as:

- Identifier
- Numerical feature
- Categorical feature
- Possible target

For example:

```text
customer_id   → identifier
age           → numerical_feature
income        → numerical_feature
contract_type → categorical_feature
tenure        → numerical_feature
churn         → possible_target
```

Identifier columns are excluded from inappropriate analytical operations such as correlations and feature visualisations.

### Statistical Analysis

The engine automatically computes analytical statistics including:

- Numerical summaries
- Categorical summaries
- Correlation matrices

Numerical statistics include:

- Count
- Mean
- Standard deviation
- Minimum
- Quartiles
- Median
- Maximum

### Intelligent EDA Planning

Instead of blindly generating every possible chart, the EDA planner creates an analysis plan based on:

- Data types
- Semantic column roles
- Cardinality
- Available numerical features
- Possible target columns

Example task:

```python
{
    "tool": "histogram",
    "column": "age",
    "reason": "Inspect the distribution of 'age'."
}
```

This separates **planning** from **execution**.

### Tool-Based EDA Execution

The executor translates planned analytical tasks into actual tool calls.

Supported visualisations include:

- Histograms
- Boxplots
- Bar charts
- Scatter plots
- Correlation heatmaps

Failures are captured individually so that one unsuccessful task does not terminate the entire analytics workflow.

### Target Analysis

Possible target variables identified during semantic analysis receive additional analysis.

For categorical targets, the engine can calculate class distributions such as:

```text
churn
Yes → 4
No  → 4
```

This prepares the architecture for future predictive analytics capabilities.

### AI-Generated Insights

After Python tools compute the analytical results, a local LLM interprets the structured evidence.

The insight generator produces:

- Overall analytical summary
- Key insights
- Evidence supporting each insight
- Importance level
- Target-specific observations
- Data cautions

The LLM is instructed to avoid inventing statistics and to distinguish association from causation.

### Automated Analytics Report

The workflow generates a Markdown report containing:

- Executive summary
- Dataset overview
- Data-quality findings
- Cleaning results
- Semantic analysis
- Statistical analysis
- Target analysis
- Key insights
- Evidence
- Data cautions
- Generated visualisations

Reports are generated under:

```text
outputs/reports/
```

### Streamlit Interface

A Streamlit interface provides an interactive way to run the analytics engine without manually executing individual Python modules.

Users can upload a CSV dataset and run the complete workflow from the browser.

---

## LangGraph Workflow

The system uses LangGraph to orchestrate the analytics pipeline.

```text
START
  │
  ▼
load_dataset
  │
  ▼
profile_dataset
  │
  ▼
data_quality
  │
  ├──── clean ────► clean_dataset
  │                    │
  │                    ▼
  │              validate_cleaning
  │                    │
  └──── skip ──────────┤
                       ▼
               semantic_analysis
                       │
                       ▼
                analysis_dataset
                       │
                       ▼
                    plan_eda
                       │
                       ▼
                  execute_eda
                       │
                       ▼
               generate_insights
                       │
                       ▼
                generate_report
                       │
                       ▼
                      END
```

Conditional routing allows the workflow to skip unnecessary cleaning when supported quality issues are not detected.

---

## Project Structure

```text
Agentic-Analytics-Engine/
│
├── data/
│   └── samples/
│       └── dirty_customers.csv
│
├── outputs/
│   ├── charts/
│   └── reports/
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── graph.py
│   │   ├── insight_generator.py
│   │   ├── nodes.py
│   │   ├── planner.py
│   │   ├── report_generator.py
│   │   ├── semantic_analyzer.py
│   │   └── state.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── cleaning.py
│       ├── data_loader.py
│       ├── data_quality.py
│       ├── profiler.py
│       ├── statistics.py
│       └── visualization.py
│
├── tests/
│   ├── test_executor.py
│   ├── test_insight_generator.py
│   ├── test_llm.py
│   ├── test_planner.py
│   ├── test_report_generator.py
│   └── test_semantic_analyzer.py
│
├── ui/
│   └── streamlit_app.py
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

## Tech Stack

### Core

- Python
- Pandas
- NumPy

### Agentic Workflow

- LangGraph

### Local AI

- Ollama
- Qwen3

Default model:

```text
qwen3:4b
```

### Visualisation

- Matplotlib
- Seaborn

### User Interface

- Streamlit

### Other

- Requests
- JSON
- TypedDict

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Blvackk/Agentic-Analytics-Engine.git
cd Agentic-Analytics-Engine
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

The project currently uses a locally running Ollama model.

Install Ollama and make sure the service is running.

Pull the default model:

```bash
ollama pull qwen3:4b
```

Verify that the model is available:

```bash
ollama list
```

The application expects Ollama at:

```text
http://localhost:11434
```

---

## Running the Application

### Streamlit UI

From the project root:

```bash
streamlit run ui/streamlit_app.py
```

Then open the local address shown by Streamlit, typically:

```text
http://localhost:8501
```

Upload a CSV dataset and run the analytics workflow.

### Command-Line Workflow

The complete workflow can also be executed using:

```bash
python app.py
```

---

## Running Tests

Individual components can be tested independently.

```bash
python -m tests.test_llm
python -m tests.test_semantic_analyzer
python -m tests.test_planner
python -m tests.test_executor
python -m tests.test_insight_generator
python -m tests.test_report_generator
```

The tests validate important behaviour including:

- LLM communication
- Structured JSON generation
- Identifier detection
- Target detection
- Identifier exclusion
- EDA planning
- EDA execution
- Target analysis
- Insight generation
- Report generation

---

## Design Principles

### Tools Calculate, LLM Interprets

Raw analytical calculations are handled by deterministic Python tools.

The LLM receives structured results and focuses on interpretation rather than performing numerical computation itself.

### Semantic Awareness

The engine distinguishes between the physical data type of a column and its analytical role.

For example, an integer `customer_id` column should not automatically be treated as a numerical feature.

### Planning and Execution Are Separate

The planner determines **what analysis should be performed**, while the executor determines **how to perform it**.

This architecture makes it easier to extend the engine with additional tools.

### Evidence-Grounded Insights

Generated insights are based on statistics calculated by the analytics pipeline.

The LLM is instructed not to fabricate statistics, causal relationships, or unsupported business conclusions.

### Failure Isolation

An individual visualisation or analytical task can fail without terminating the entire workflow.

Errors are recorded in the workflow state for inspection.

---

## Current Limitations

Phase 1 currently focuses on automated exploratory data analysis.

Some limitations include:

- Primarily designed for CSV datasets
- Local Ollama dependency
- Semantic interpretation depends partly on LLM quality
- Automatic cleaning intentionally supports a limited set of safe transformations
- Small datasets can produce statistically unreliable relationships
- Correlation analysis does not establish causation
- No predictive ML pipeline yet
- No conversational dataset querying yet

---

## Roadmap

### Phase 1 — Automated Analytics Engine

- [X] Dataset loading
- [X] Dataset profiling
- [X] Data-quality detection
- [X] Conditional cleaning
- [X] Cleaning validation
- [X] Semantic column analysis
- [X] Statistical analysis
- [X] EDA planning
- [X] Tool execution
- [X] Automated visualisation
- [X] Target analysis
- [X] LLM-generated insights
- [X] Markdown report generation
- [X] LangGraph orchestration
- [X] Streamlit interface

### Phase 2 — Conversational Analytics

Planned capabilities:

- [ ] Chat with uploaded datasets
- [ ] Natural-language query understanding
- [ ] Query-specific analytical planning
- [ ] Dynamic tool selection
- [ ] Query-specific statistical execution
- [ ] Evidence-grounded answers
- [ ] Follow-up questions
- [ ] Conversation context
- [ ] Interactive analytics through Streamlit

### Future Extensions

Potential later additions include:

- Automated ML problem detection
- Feature engineering
- Model selection and evaluation
- MLflow experiment tracking
- Forecasting workflows
- Anomaly detection
- SQL/database support
- Excel support
- Exportable PDF reports
- Human approval checkpoints
- Deployment

---

## Example

Given a customer dataset containing:

```text
customer_id
age
income
contract_type
tenure
churn
```

the semantic analyzer may determine:

```text
customer_id   → Identifier
age           → Numerical Feature
income        → Numerical Feature
contract_type → Categorical Feature
tenure        → Numerical Feature
churn         → Possible Target
```

The planner can then exclude `customer_id` from feature analysis while selecting appropriate statistical and visual analyses for the remaining columns.

The resulting statistics are passed to the LLM, which generates evidence-grounded interpretations and cautions that are incorporated into the final report.

---

## Repository

GitHub:

https://github.com/Blvackk/Agentic-Analytics-Engine

---

## Status

**Phase 1: Complete**

The automated analytics workflow is operational end-to-end.

Development is continuing with **Phase 2: Conversational Analytics / Chat with Data**.
