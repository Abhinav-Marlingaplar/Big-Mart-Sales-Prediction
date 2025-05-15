# Big Mart Sales Prediction using XGBoost

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%2CNumPy%2CXGBoost%2CSklearn%2CMatplotlib%2CSeaborn-brightgreen.svg)

## Overview

This project focuses on developing a predictive model to forecast sales for different items across Big Mart outlets. By leveraging historical sales data and various product and store attributes, the goal is to build an accurate and reliable sales prediction system. The project utilizes the XGBoost Regressor algorithm, a powerful and efficient gradient boosting framework, to achieve this objective.

## Key Features

* **Data Exploration and Preprocessing:** Comprehensive analysis of the dataset, including handling missing values and visualizing key features.
* **Feature Engineering:** Transformation of categorical variables into numerical representations using Label Encoding to make them suitable for the machine learning model.
* **Robust Model:** Implementation of the XGBoost Regressor, known for its high performance in regression tasks.
* **Overfitting Prevention:** Utilization of early stopping during model training to mitigate overfitting and ensure good generalization on unseen data.
* **Performance Evaluation:** Thorough evaluation of the model's predictive accuracy using the R-squared ($R^2$) metric.

## Technologies Used

* **Python:** Programming language used for the entire project.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical computations.
* **XGBoost:** The gradient boosting library used for the regression model.
* **Scikit-learn (sklearn):** For data splitting, preprocessing (Label Encoding), and model evaluation metrics.
* **Matplotlib and Seaborn:** For creating insightful data visualizations.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    https://github.com/Abhinav-Marlingaplar/Big-Mart-Sales-Prediction.git
    cd Big-Mart-Sales-Prediction
    ```

2.  **Install required libraries manually:**

    This project relies on the following Python libraries. Please ensure you have them installed in your environment. You can install them using pip:

    ```bash
    pip install pandas numpy xgboost scikit-learn matplotlib seaborn jupyter
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook BigMartSalesPrediction.ipynb
    ```

## Usage

The Jupyter Notebook (`BigMartSalesPrediction.ipynb`) contains the complete end-to-end workflow of the project, including data loading, preprocessing, model training, and evaluation. Simply run the cells in the notebook sequentially to reproduce the results.

## Results

The final evaluation of the XGBoost Regressor model on the test dataset yielded an R-squared ($R^2$) score of approximately **0.614**. This indicates that the model explains about 61.4% of the variance in the Big Mart sales based on the features provided. While this represents a reasonable level of predictive power, further efforts in feature engineering and hyperparameter tuning could potentially lead to improved accuracy. The model also achieved an R-squared score of approximately 0.632 on the training dataset.

## Future Work

* Explore additional feature engineering techniques to potentially improve model accuracy.
* Perform hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV to optimize the XGBoost model.
* Experiment with other regression algorithms and compare their performance.
* Investigate the impact of different data scaling methods.
* Consider deploying the model for real-time sales forecasting.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

Abhinav Marlingaplar
