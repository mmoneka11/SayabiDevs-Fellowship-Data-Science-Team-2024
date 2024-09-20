# Heart Disease Prediction Project

This project focuses on building and deploying a machine learning model to predict heart disease using the Heart Disease dataset from the UCI Machine Learning Repository. The model classifies the presence or absence of heart disease based on patient attributes.

## Dataset Information

- **Name**: Heart Disease Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Data URL**: [Dataset CSV](https://archive.ics.uci.edu/static/public/45/data.csv)
- **Abstract**: The dataset consists of data collected from four different sources: Cleveland, Hungary, Switzerland, and the VA Long Beach. Only the Cleveland database is used, as it has been processed to remove missing values.

## Project Workflow

1. **Data Preprocessing**: 
   - Handled missing values.
   - Normalized or scaled numerical features.

2. **Exploratory Data Analysis**:
   - Visualized data distributions and relationships.
   - Identified key features influencing heart disease.

3. **Model Development**:
   - Several machine learning models were evaluated during the development of this project.Then models were compared based on accuracy, precision, recall, and F1-score. The best-performing model was chosen based on these metrics.
   - We utilize the following models:
   ```bash
        Logistic Regression
        Decision Tree
        Random Forest
        Support Vector Machine (SVM)
        K-Nearest Neighbors (KNN)
        Gradient Boosting
        XGBoost
        AdaBoost
        Naive Bayes
        LightGBM
   ```

4. **Evaluation and Tuning**:
   - The model's performance is evaluated using the following metrics:
      - **Accuracy**: The overall correctness of the model's predictions.
      - **Precision**: The proportion of positive identifications that were actually correct.
      - **Recall**: The proportion of actual positives that were identified correctly.
      - **F1 Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
   - Evaluated the model using cross-validation and confusion matrix. Fine-tune the hyperparameters for optimal performance.

6. **Deployment**:
   - Deploy the trained model as a web service or via a cloud-based API (optional).
  
7. **Conclusion**:
  The chosen machine learning model successfully predicts heart disease with high accuracy, showing that it can be a valuable tool for healthcare professionals to identify at-risk individuals.

## Best Model
   - The best performing model is selected based on K-Fold cross-validation accuracy and other evaluation metrics
   - .Naive Bayes is the best model for this particular dataset due to its high accuracy, stability, and effective handling of class predictions.
   - The Naive Bayes model achieved the highest cross-validated accuracy with 83.61% and KFold accuracy of 83.08%.
 ### Results

     The final model achieved the following results:
      
      - **Accuracy**: 84%
      - **Precision**: 84%
      - **Recall**: 84%
      - **F1 Score**: 84%
   - Naive Bayes achieved an accuracy of 0.84, which is the highest among all models tested. This indicates that it correctly predicted the outcome more often than any other model.The model also has a relatively low standard deviation (± 0.0448) during cross-validation, suggesting that its performance is stable across different subsets of the data. This consistency is crucial for generalization to unseen data.


## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
    git clone https://github.com/mmoneka11/SayabiDevs-Fellowship-Data-Science-Team-2024.git   
  ```

2. Navigate to the project directory:
   ``` bash
      cd Heart_Disease_Prediction
   ```

3. Create a virtual environment and install dependencies:
   ``` bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     pip install -r requirements.txt
   ```
   
4. Run the Jupyter Notebook:

   ```bash
    jupyter notebook
   ```
## Dependencies
   ```bash
    Python 3.8+
    Jupyter Notebook
    Scikit-learn
    Pandas
    NumPy
    Matplotlib
    Seaborn
  ```

## Contribution of Each Member:

| **Name** | **Responsibility**                              |
|----------|-------------------------------------------------|
| Hassan   | Data Collection and Exploration                 |
| Moneka   | Data Preprocessing, Model Development, and Comparison |
| Fahad    | Model Evaluation                                |
| Malaika  |                                                 |
| Shaeel   |                                                 |

## Authors' Profiles

* **Moneka**:
  
   [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/mmoneka11)
   [![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/mmoneka11)
   [![Medium](https://img.shields.io/badge/Medium-Blog-blue?style=for-the-badge&logo=medium)](https://medium.com/@mmoneka11)
   [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mmoneka11/)
   [![Twitter](https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter)](https://twitter.com/mmoneka11)

* **Hassan**:
  
   [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/HassanSharif7)
   [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/hassan-sharif-301672257)

* **Shaeel**:
  
   [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/shaeel24)
   [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://pk.linkedin.com/in/shaeel-khalil-060aa2241)

* **Fahad**:
  
   [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/21k-3103-Fahad/)

* **Malaika**:

   [![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/malayeka971)


## Acknowledgments

This project is part of the SayabiDevs Fellowship Data Science Team 2024. Special thanks to the mentors and team members for their guidance and support.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
