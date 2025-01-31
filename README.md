# Pokémon Dataset Analysis and Prediction

## 📌 Project Overview

This project focuses on analyzing a Pokémon dataset by performing essential data preprocessing, exploratory data analysis (EDA), handling missing values, detecting and treating outliers, and training a linear regression model to predict the total stats of Pokémon based on various features.

## 🛠️ Features

- **Data Preprocessing**: Handles missing values, removes redundant columns, and applies one-hot encoding.
- **Exploratory Data Analysis (EDA)**: Visualizes data distributions using box plots.
- **Outlier Detection & Handling**: Uses the interquartile range (IQR) method and winsorization to treat outliers.
- **Machine Learning**: Trains and evaluates a linear regression model to predict Pokémon's total stats.
- **Data Visualization**: Generates meaningful plots to understand the dataset better.

## 📂 Dataset Information

The dataset contains various attributes for each Pokémon, including:

- **id**: Unique identifier for each Pokémon.
- **name**: Pokémon name.
- **type\_1, type\_2**: Primary and secondary Pokémon types.
- **total**: Sum of all base stats.
- **hp**: Hit points (health points).
- **attack**: Attack power.
- **defense**: Defensive ability.
- **special\_attack**: Special attack power.
- **special\_defense**: Special defense ability.
- **speed**: Pokémon's speed.
- **generation**: Pokémon generation.
- **is\_legendary**: Whether the Pokémon is legendary or not.

## 🔧 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pokemon-analysis.git
   cd pokemon-analysis
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your dataset (`dataset.csv`) is placed in the project folder.
4. Run the script:
   ```bash
   python main.py
   ```

## 📊 Data Processing Steps

1. **Loading Data**: Reads `dataset.csv` into a Pandas DataFrame.
2. **Handling Missing Values**:
   - Uses mode for categorical data.
   - Uses median for numerical attributes.
3. **Dropping Unnecessary Columns**:
   - Removes `id`, `name`, and `type_2`.
4. **Outlier Detection & Handling**:
   - Uses IQR to identify outliers.
   - Applies winsorization to limit extreme values.
5. **Encoding Categorical Features**:
   - Uses one-hot encoding for `type_1` and `generation`.
6. **Data Splitting**:
   - Splits dataset into 80% training and 20% testing.

## 📈 Machine Learning Model

A linear regression model is trained to predict the `total` stat of a Pokémon based on its other attributes.

### Model Training

- Uses `sklearn.linear_model.LinearRegression`.
- Fits the model using the training dataset.

### Model Evaluation Metrics

- **Mean Squared Error (MSE)**: Measures prediction error.
- **R² Score**: Indicates the model's explanatory power.

### Visualizing Predictions

- Generates scatter plots comparing actual and predicted values.

## 🔍 Example Output

```
***** Evaluate Predictions on Training Set *****
Mean Squared Error (MSE) = 124.56
Coefficient of Determination (R² score) = 0.87

***** Evaluate Predictions on Test Set *****
Mean Squared Error (MSE) = 140.21
Coefficient of Determination (R² score) = 0.85
```

## 📊 Data Visualization

- Box plots are created for each numerical column.
- Scatter plots compare actual vs. predicted values.

## 📝 Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

