# Localization and Classification Model

This project aims to develop and compare various classification models for localization tasks. The models implemented are Support Vector Classifier (SVC), Naive Bayes, Logistic Regression, and Artificial Neural Networks (ANN). The sliding window approach is used for localization.

## Project Setup

This project uses a Python virtual environment (`venv`). Follow the instructions below to set up your environment and run the project.

### Prerequisites

- Python 3.x

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Create and activate the virtual environment:**

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


**Install the required dependencies:**
pip install -r requirements.txt

Ensure that the following packages are listed in requirements.txt and installed:
scikit-learn
numpy
joblib
pandas
tensorflow (for ANN)
