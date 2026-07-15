# 📋 Complaint Management System - Data Processing & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project analyzes and manages customer complaint data collected from a telecom company, focusing on the effective categorization, lifecycle tracking, and resolution of customer complaints. The data provides insights into how complaints are handled and escalated within the organization, aiming to enhance efficiency in resolving issues.

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Visualizations](#visualizations)
- [Installation & Usage](#installation--usage)
- [Key Findings](#key-findings)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The goal of this project is to classify customer complaints into **Technical** or **Commercial** categories using machine learning. The pipeline includes:

1. **Data Cleaning** – Handling missing values with context-aware imputation strategies
2. **Feature Engineering** – Creating time-based features (resolution time, open hour, etc.)
3. **Data Leakage Prevention** – Identifying and removing features that directly leak the target
4. **Model Training** – Training 6 different ML models with proper cross-validation
5. **Model Comparison** – Comprehensive evaluation with multiple metrics and visualizations

## 📊 Dataset

The dataset contains **10,400+** complaint records from a telecom company with **22 columns**:

| Category | Columns |
|----------|---------|
| **Identifiers** | `CASE_ID` |
| **Customer Info** | `CUSTOMER_TYPE`, `CUSTOMER_GROUP`, `OFFER_NAME` |
| **Complaint Lifecycle** | `CURRENT_STATUS`, `OPEN_DATE`, `CLOSE_DATE`, `OPEN_USER`, `CLOSE_USER` |
| **Escalation** | `ESCALATION_FLAG`, `ESCALATED_GROUP` |
| **Groups** | `OPEN_GR`, `CLOSE_GROUP` |
| **Classification** | `COMPLAINT_TYPE` (Target), `ACTUAL_COMPLAINT`, `CALLBACK_MECHANISM` |
| **Details** | `AGE_BRACKET`, `RESOLUTION`, `RESOLUTION_DESCRIPTION`, `CASE_DESC` |

> **Note:** The raw data is not included in this repository due to privacy considerations. See [data/README.md](data/README.md) for the schema details.

## 📁 Project Structure

```
Complaint-Management-System-Data-Processing/
├── 📂 data/
│   └── README.md           # Dataset schema & instructions
├── 📂 Notebook/
│   └── Final.ipynb         # Original Jupyter notebook (exploratory)
├── 📂 src/
│   ├── __init__.py
│   ├── main.py             # Main pipeline entry point
│   ├── data_processing.py  # Data loading, cleaning & feature engineering
│   ├── model_training.py   # Model training & cross-validation evaluation
│   └── visualization.py    # Professional charts & visualizations
├── 📂 outputs/             # Generated charts & results (after running)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🔬 Methodology

### Data Cleaning

| Issue | Strategy |
|-------|----------|
| Columns with >85% missing (`RESOLUTION`, `RESOLUTION_DESCRIPTION`, `CASE_DESC`) | Dropped |
| `CLOSE_USER` missing | Filled with `OPEN_USER` (assumption: same user closes) |
| `CLOSE_DATE` missing | Rows dropped (required for time features) |
| `ESCALATED_GROUP` missing | Filled with `"NO_GROUP"` (not escalated) |
| `CALLBACK_MECHANISM` missing | Filled with mode (most frequent) |
| `OFFER_NAME` / `CUSTOMER_GROUP` | Context-aware imputation based on relationships |
| `OPEN_GR` / `CLOSE_GROUP` | Cross-reference frequency-based imputation |

### Data Leakage Prevention

> ⚠️ The columns `CASE` and `PRODUCT` were identified as **data-leaky features** – they are directly derived from `COMPLAINT_TYPE` (e.g., "Technical complaint" always maps to `COMPLAINT_TYPE = Technical`). These columns were removed to get realistic model performance.

### Feature Engineering

New features created from datetime columns:
- `RESOLUTION_TIME_HOURS` – Time to resolve the complaint
- `OPEN_HOUR` – Hour of the day the complaint was opened
- `OPEN_DAY_OF_WEEK` – Day of the week
- `OPEN_MONTH` – Month of the year

### Feature Scaling

All features are standardized using `StandardScaler` before training (critical for KNN, SVM, and Logistic Regression).

## 🤖 Models & Results

Six classification models were evaluated using **10-Fold Stratified Cross Validation**:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| K-Nearest Neighbors (KNN) | - | - | - | - |
| Decision Tree | - | - | - | - |
| Gaussian Naive Bayes | - | - | - | - |
| **Random Forest** | - | - | - | - |
| SVM | - | - | - | - |
| Logistic Regression | - | - | - | - |

> 📝 **Note:** Run the pipeline to generate actual results. Values depend on the dataset and will be saved to `outputs/model_results.csv`.

## 📈 Visualizations

The pipeline generates the following visualizations in the `outputs/` directory:

### EDA Charts
- **Complaint Type Distribution** – Bar chart & pie chart of target variable
- **Escalation Analysis** – Escalation patterns by complaint type
- **Actual Complaint Distribution** – Real vs. non-actual complaints
- **Callback Mechanism** – Customer contact methods
- **Customer Type Distribution** – CBU/EBU breakdown

### Model Comparison Charts
- **Model Comparison Dashboard** – Side-by-side metric comparison
- **Radar Chart** – Multi-metric spider plot for all models
- **Confusion Matrices** – Grid of confusion matrices for each model
- **Performance Heatmap** – Color-coded metric comparison

## 🚀 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/Saif-Damra/Complaint-Management-System-Data-Processing.git
cd Complaint-Management-System-Data-Processing

# Install dependencies
pip install -r requirements.txt

# Place your data file
# Copy Complaints.csv to the data/ directory
```

### Running the Pipeline

```bash
# Run with default settings
python src/main.py

# Specify custom data path
python src/main.py --data path/to/Complaints.csv

# Specify output directory
python src/main.py --output results/

# Custom number of CV folds
python src/main.py --folds 5

# Skip EDA visualizations
python src/main.py --skip-eda
```

### Running the Notebook

```bash
jupyter notebook Notebook/Final.ipynb
```

## 🔑 Key Findings

1. **Data Leakage was inflating accuracy** – The original notebook achieved 99.5% accuracy on Decision Tree, but this was due to leaky features (`CASE`, `PRODUCT`) that directly encode the target variable.

2. **Feature Scaling matters** – KNN performance improved significantly after applying `StandardScaler`.

3. **Cross-Validation provides more reliable estimates** – Using `StratifiedKFold` instead of random train/test splits gives more stable and trustworthy results.

4. **Context-aware imputation** – Missing values in `OFFER_NAME` and `CUSTOMER_GROUP` were filled based on known business relationships (e.g., `FTTH Home` offer always belongs to `FTTH Home` customer group).

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by [Saif Damra](https://github.com/Saif-Damra)**
