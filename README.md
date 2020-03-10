# Multiple-Linear-Regression

We will be solving a simple business task using Multiple Linear Regression.

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
New York has the highest p-value (0.990 > 0.05) so that variable will be removed, we will run Reg_OLS and review the metrics

```
X_opt = X[:,[0,1,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```
Output:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     217.2
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           8.49e-29
Time:                        21:46:24   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1061.
Df Residuals:                      45   BIC:                             1070.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
Omnibus:                       14.758   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.172
Skew:                          -0.948   Prob(JB):                     2.53e-05
Kurtosis:                       5.563   Cond. No.                     1.40e+06
==============================================================================
```

Florida has the highest p-value (0.940 > 0.05) so that variable as been removed, we will run Reg_OLS and review the metrics
```
X_opt = X[:,[0,3,4,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```

Output:

```
                             OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     296.0
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           4.53e-30
Time:                        21:50:39   Log-Likelihood:                -525.39
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      46   BIC:                             1066.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================
```

Administration has the highest p-value (0.602 > 0.05) so that variable as been removed, we will run Reg_OLS and review the metrics

```
X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           2.16e-31
Time:                        21:53:12   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================
```
Marketing spend has the highest p-value (0.06 > 0.05)  so that variable as been removed, we will run Reg_OLS and review the metrics

```
X_opt = X[:,[0,3]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```
Output:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           3.50e-32
Time:                        21:55:06   Log-Likelihood:                -527.44
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      48   BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================
```
As the adjusted R-squared has decreased, we will add the "Marketing Spend" predictor back into our model and declare this our optimal multiple linear regression model. We conclude that we should invest in companies that are investing highly in "R&D" (Research and Development), with "Marketing Spend" being the additional (but weaker) contributing factor.

```
X_opt = X[:,[0,3,5]]
Reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(Reg_OLS.summary())
```

Output:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Mon, 09 Mar 2020   Prob (F-statistic):           2.16e-31
Time:                        21:57:34   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================
```
where X1 is R&D Spend and X2 is Marketing Spend.

Our regressor therefore takes the following function:

```
Profit = 46980 + (0.7966)X1 + (0.0299)X2
```
**Note:**

It is incorrect to make statements such as "R&D spend has a much larger impact on profit in comparison to Marketing spend" or "R&D's impact on profit is over 26 times more than Marketing spend". This is because, solely looking at the coefficients does not give us the unit measurements for theses predictors. For instance, if R&D spend was in £10,0000's and Marketing spend was is in £1,000's, the amount invested in each will differ substantially. Therefore, we conclude with the below statement:

R&D spend has a greater impact on profit per unit of R&D spend than Marketing Spend has per unit of marketing spend. For every one unit increase/decrease in R&D spend, the Profit will increase/decrease by 0.7966 unit pounds.
