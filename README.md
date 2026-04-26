# Energy Efficiency Optimization using Gradient Boosting & SHAP

## Overview
This project predicts the **Heating Load** of buildings based on architectural and structural parameters. Using the Energy Efficiency dataset from the UCI Machine Learning Repository, I developed a machine learning pipeline that achieves high predictive accuracy while maintaining full model transparency through Explainable AI (XAI) techniques.

## Key Features
- **End-to-End Pipeline**: Includes data preprocessing, feature scaling, and rigorous cross-validation to prevent data leakage.
- **Model Selection**: Comparative analysis between Linear Models and Ensemble Methods (Random Forest, Gradient Boosting).
- **Optimization**: Hyperparameter tuning using Grid Search to reach an R² score of ~0.99.
- **Interpretability (XAI)**: Integration of **SHAP values** to decode the black box model and identify how features like 'Relative Compactness' and 'Overall Height' influence energy demand.

## Tech Stack
- **Language**: Python
- **Libraries**: Pandas, Scikit-Learn, SHAP, Matplotlib, Seaborn
- **Environment**: Google Colab / Jupyter Notebook

## Results
The optimized **Gradient Boosting Regressor** significantly outperformed the baseline linear model, reducing the RMSE from 2.9 to 0.4. SHAP analysis revealed that building compactness and height are the most critical factors, providing actionable insights for sustainable architectural design.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas scikit-learn shap matplotlib seaborn`.
3. Open and run the `Energy_Efficiency.ipynb` notebook.
