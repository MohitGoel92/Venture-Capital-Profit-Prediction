# Multiple-Linear-Regression

We will explore be solving a simple business task using Multiple Linear Regression.

**Task: Venture capital fund analysis** 

We have been given a dataset that contains a list of 50 companies with their yearly expenditure on R&D (Research & Development), Administration, Marketing and State (New York, California, Florida), alongside their Profit. Our task is to explore which companies we should invest in in order to optimise our profits.

We will be using Multiple Linear Regression to carry out this task.

**Assumptions of a linear regression:**
- Linearity
- Homoscedasticity
- Multivariate Normality
- Independence of errors
- Lack of multicollinearity

**Backwards Elimination:** In order to build an "optimal" multiple linear regression model, we use backwards elimination to find the optimal number of independent variables so that each variable has a significant impact on the dependent varaible (profit). In our case, we are using a 5% significance level (p-value = 0.05), therefore any predictor variable that has a p-value > 0.05 should be removed, and we will run the Regressor Ordinary Least Squares (Reg_OLS) again and observe the metrics thereafter.
