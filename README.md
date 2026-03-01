
# Heart Failure Prediction Model

A machine learning model to predict heart disease events, optimized for high recall in medical diagnosis.

## Table of Contents

- [Project Title](#project-title)
- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## About The Project

This project develops and optimizes a machine learning model to predict heart disease events. Initial efforts focused on addressing class imbalance using SMOTE and evaluating a RandomForestClassifier. Subsequent optimizations involved exploring polynomial features, experimenting with a Gradient Boosting Classifier, performing rigorous hyperparameter tuning, and analyzing misclassified samples to improve critical metrics like recall. The final recommended model is a **Gradient Boosting Classifier** configured for optimal performance in a medical context where minimizing false negatives is paramount.

### Built With

List the major frameworks/libraries/technologies used in your project.
* Python
* Pandas
* Scikit-learn
* Matplotlib
* imbalanced-learn
* Joblib
* Streamlit

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Ensure you have Python installed. This project also requires several Python libraries which can be installed via pip.

*   pip

### Installation

1.  Clone the repo
    ```bash
    git clone https://github.com/your_username/your_project.git
    ```
2.  Navigate to the project directory
    ```bash
    cd your_project
    ```
3.  Install Python packages
    ```bash
    pip install pandas scikit-learn matplotlib imbalanced-learn joblib streamlit
    ```
4.  Place your `heart.csv` dataset, the trained model `model.pkl`, and the `polynomial_features_transformer.pkl` file in the project root directory.

## Usage

To run the analysis and deploy the prediction application:

### Running the Notebook Analysis:
Open and run the Jupyter notebook (`your_notebook_name.ipynb`). This will load the data, split it, perform data balancing, implement feature engineering, train and tune models, and evaluate performance. The notebook demonstrates the entire optimization process.

### Running the Streamlit Application (for prediction):

After ensuring `model.pkl` and `polynomial_features_transformer.pkl` are in the same directory as `app.py`:

```bash
streamlit run app.py
```

This will launch a web application where you can input patient parameters and get predictions. The application uses the **Gradient Boosting Classifier** trained with polynomial features and applies an **adjusted classification threshold of 0.25** to prioritize recall, which is critical for identifying heart disease events.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Albert Antony S - [albertantony1820@gmail.com](mailto:albertantony1820@gmail.com)

Project Link: [Heart-Failure-Prediction-using-Gradient-Boosting-Classifier](https://github.com/AlbertAntony20/Heart-Failure-Prediction-using-Gradient-Boosting-Classifier/tree/main)
