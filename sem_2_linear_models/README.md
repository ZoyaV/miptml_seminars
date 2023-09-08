# README: Linear Regression
## Table of Contents

- [1. Theory]
- [2. Gradient Descent and Other Linear Models Optimization Techniques]
- [3. Metrics]
- [4. Considerations When Implementing]
- [5. Analyzing Feature Influence using Linear Regression]

## 1. Theory:
Linear regression models the relationship between a dependent variable and 
one or more independent variables using a linear equation. The 
relationship is represented in matrix notation as:

\[ Y = X \beta + \epsilon \]

Where:
- \( Y \) is a vector of the dependent variable.
- \( X \) is the design matrix of independent variables, with an added 
column of ones to account for the intercept.
- \( \beta \) is a vector of coefficients.
- \( \epsilon \) is the error term.

The goal is to find the \( \beta \) that minimizes the sum of squared 
residuals. This is achieved using:

\[ \beta = (X^T X)^{-1} X^T Y \]

Implementation with sklearn
```python
from sklearn.linear_model import LinearRegression

# Create an instance of the model
regressor = LinearRegression()

# Fit the model to data
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)
```

Ensure you've already divided your data into training and test sets (e.g., 
using `train_test_split` from `sklearn`) and imported necessary libraries 
before executing the above code.

## 2. Gradient Descent and Other Linear Models Optimization Techniques:

When dealing with large datasets, the matrix operations required for linear regression (like matrix inversion) become computationally intensive and impractical. To overcome these challenges, optimization techniques like gradient descent are employed.

##### Gradient Descent:

Gradient Descent is an iterative optimization algorithm used to minimize the cost function. The idea is to update the coefficients of the model in the direction that decreases the cost function the most, until the cost function converges to a minimum value.

**Steps**:
1. Initialize the coefficients with random values.
2. Calculate the gradient of the cost function concerning each coefficient.
3. Update the coefficients in the direction opposite to the gradient.
4. Repeat steps 2 and 3 until convergence.

The size of the steps taken towards the minimum is determined by the learning rate. A smaller learning rate could be slower but more precise, while a larger learning rate could lead to overshooting the minimum.

##### Variations of Gradient Descent:
1. **Batch Gradient Descent**: Uses the entire dataset to compute the gradient and update the coefficients in each iteration.
2. **Stochastic Gradient Descent (SGD)**: Updates the coefficients based on each data point, making it faster but more erratic.
3. **Mini-Batch Gradient Descent**: A middle-ground approach where mini-batches of data are used to update coefficients.

##### Regularized Linear Models:
When using gradient descent, it's common to incorporate regularization to prevent overfitting.

1. **Ridge Regression (L2 regularization)**: Penalizes the squared magnitude of coefficients.
2. **Lasso Regression (L1 regularization)**: Penalizes the absolute value of coefficients, leading to some coefficients becoming exactly zero, thus performing feature selection.

##### Implementation:

Many libraries, including `scikit-learn`, offer linear models optimized using gradient descent:

```python
from sklearn.linear_model import SGDRegressor

# Create an instance of the SGDRegressor
regressor = SGDRegressor(penalty='none', eta0=0.01, max_iter=1000, tol=1e-3)

# Fit the model
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)
```

Note: When using gradient descent-based methods, it's crucial to scale or normalize the input features.
## 3. Metrics

 Let's briefly discuss the regression metrics in English.

**1. Mean Absolute Error (MAE)**
\[ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \]

**Features**:
- Easy to interpret; average absolute prediction error.
- Doesn't account for the direction of errors.
- Less sensitive to outliers than MSE.

**When to use**: When you want your model to be penalized linearly for prediction errors.

---

**2. Mean Squared Error (MSE)**
\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

**Features**:
- Strongly penalizes larger errors.
- Sensitive to outliers.

**When to use**: When you want to particularly minimize large errors.

---

**3. Root Mean Squared Error (RMSE)**
\[ RMSE = \sqrt{MSE} \]

**Features**:
- Error in the same units as the data.
- Penalizes large errors.

**When to use**: When units matter and large errors are critical.

---

**4. Mean Absolute Percentage Error (MAPE)**
\[ MAPE = \frac{100%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \]

**Features**:
- Expresses error as a percentage of true values.
- Undefined for true values of zero.
- Can inflate errors for small true values.

**When to use**: When expressing error as a percentage is desirable, but be cautious with datasets having zero or very small true values.

---

**5. Coefficient of Determination \( R^2 \)**
\[ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \]

**Features**:
- Represents the proportion of variance explained by the model.
- Ranges from -∞ to 1. A value of 1 indicates a perfect fit; 0 indicates the model predicts no better than the mean.

**When to use**: When you want to evaluate how much variance your model explains compared to just using the mean value.

## 4. Considerations When Implementing:
- **Assumptions**: Ensure that the assumptions of linear regression are 
met, which include linearity, independence, homoscedasticity, and 
normality.
- **Multicollinearity**: If the independent variables are highly 
correlated, it can destabilize the estimation of coefficients. Techniques 
like VIF (Variance Inflation Factor) can be used to detect it.
- **Feature Scaling**: It’s often a good practice to scale or normalize 
the data to ensure convergence while using gradient-based optimization 
methods.
- **Outliers**: Outliers can have a disproportionate impact on the 
regression line. Consider outlier detection techniques.
- **Overfitting**: If the model has too many predictors or is too complex, 
it can overfit to the training data.


## 5. Analyzing Feature Influence using Linear Regression:

Linear regression can be instrumental in assessing the impact of individual features on the dependent variable. The coefficients associated with each feature indicate the change in the dependent variable for a one-unit change in the predictor, assuming other predictors remain constant.

In Python, the `statsmodels` library is often used to get a comprehensive summary of the regression, which includes coefficients, t-values, p-values, and other statistics which can be used to infer the significance and effect of each feature.

#### Example:

Let's consider a simple scenario where we are trying to predict the price of houses (`price`) based on their size (`size`) and the number of bedrooms (`bedrooms`).

Using `statsmodels`:

```python
import statsmodels.api as sm
import numpy as np

# Let's assume you have a DataFrame `house_data` with columns 'price', 'size', and 'bedrooms'
X = house_data[['size', 'bedrooms']]
X = sm.add_constant(X)  # Adds a constant column for intercept
y = np.log(house_data['price'])  # We take the log of price for normalization

model = sm.OLS(y, X)
fitted = model.fit()
print(fitted.summary())
```

In the summary, the coefficient for each variable shows its influence on the `price`. A positive coefficient indicates that as the feature value increases, the `price` will also increase, and vice versa. The p-value associated with each coefficient can be used to test the hypothesis that the coefficient is equal to zero (no effect). A low p-value (<0.05) indicates that you can reject the null hypothesis.

Remember to interpret the coefficients in the context of the transformation you might have applied to the dependent variable (e.g., if you took the log of the price, the interpretation will be in terms of percentage change).
