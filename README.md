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

**Backwards Elimination:** In order to build an "optimal" multiple linear regression model, we use backwards elimination to find the optimal number of independent variables so that each variable has a significant impact on the dependent varaible (profit). In our case, we are using a 5% significance level (p-value = 0.05), therefore any predictor variable that has a p-value > 0.05 should be removed and we will run the Regressor Ordinary Least Squares (Reg_OLS) again and observe the metrics thereafter.

The below code will be our first run of the Reg_OLS: 
```
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```
Output:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     169.9
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           1.34e-27
Time:                        21:43:31   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1063.
Df Residuals:                      44   BIC:                             1074.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
Omnibus:                       14.782   Durbin-Watson:                   1.283
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.266
Skew:                          -0.948   Prob(JB):                     2.41e-05
Kurtosis:                       5.572   Cond. No.                     1.45e+06
==============================================================================
```
New York has the highest p-value (0.990>0.05) so that variable will be removed, we will run Reg_OLS and review the metrics again

```
X_opt = X[:,[0,1,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```
output:


# Florida has the highest p-value (>0.05) so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Administration has the highest p-value (>0.05) so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# Marketing spend has the highest p-value (>0.05)  so that variable as been removed, we will run Reg_OLS and review the metrics again

X_opt = X[:,[0,3]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())

# As the adjusted R-squared has decreased, we will add the "Marketing Spend" variable back and declare this our multiple linear regression model
# We conclude that we should invest in companies that are investing highly in "R&D" (Research and Development), with "Marketing Spend" being the
# additional (but weaker) contributing factor

X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(Reg_OLS.summary())
