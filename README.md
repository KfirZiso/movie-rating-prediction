# 🎬 Movie Rating Prediction Model - IMDb
**Final Project - Part 2 | Machine Learning Course**

## 📝 Project and Model Description
This project presents the development and evaluation of a machine learning model to predict the average rating (`averageRating`) of movies from the IMDb database. 
The objective of the model is to make predictions **prior to the movie's theatrical release**, with a strict emphasis on preventing Data Leakage. To achieve this, features that accumulate only after the release, such as the number of votes (`numVotes`) and box office revenue (`BoxOffice`), were removed from the dataset.

**Evaluated Models:**
Extensive Feature Engineering was conducted, including extracting A-list actors, creating runtime categories (Binning), adding genre flags, and more. 
Two models were trained and evaluated using 10-Fold Cross Validation:
1. **Elastic Net:** A linear regression model with L1 and L2 regularization, integrated with Polynomial Features within the Pipeline to capture complex relationships in the data.
2. **Random Forest Regressor:** A decision tree-based model (100 trees), which excels at identifying non-linear relationships and smart splits, and is insensitive to multicollinearity.

**The Final Model:** The trained model with the best performance was saved to a `model.pkl` file (using serialization libraries). The statistical data preprocessing steps (Scaling, Imputation, One-Hot Encoding) are hermetically sealed within the model's Pipeline.

---

## 🛠️ System Requirements
The project was developed and executed in a **Python 3.10+** environment.
To run the code, ensure the following libraries are installed (as listed in the provided `requirements.txt` file):

* pandas
* numpy
* scikit-learn
* joblib
* matplotlib
* seaborn

You can install all the required libraries by running the following command in your terminal:
```bash
pip install -r requirements.txt
```

---

## 🚀 Run Instructions
To test and evaluate the model on a new Held-Out Test Set during the final presentation/defense, please follow these steps in order:

### Step 1: Import Required Libraries and Functions
Ensure that the model file (`model.pkl`) and the data processing function (`prepare_data`) defined in our notebook are loaded into the execution environment.
```python
import pandas as pd
import joblib
# Assuming the prepare_data function from the notebook is already defined/imported in the environment
```

### Step 2: Read and Process the Data
Load the new dataset file and pass it through our `prepare_data` function. This function cleans text fields and generates the engineered features without performing any statistical manipulations.
```python
# Read the new dataset
df_test = pd.read_csv('path_to_new_dataset.csv')

# Process the data using our function
X_test = prepare_data(df_test)
```

### Step 3: Load the Model and Make Predictions
Now, load the trained Pipeline from the Pickle file and execute the prediction mechanism. All statistical preprocessing of the model (Scaling, Imputation) will automatically be applied to `X_test` behind the scenes, based on the calculations from the training set.
```python
# Load the model from the file
model = joblib.load('model.pkl')

# Generate predictions
y_pred = model.predict(X_test)

# Print the predictions
print(y_pred)
```