import pandas as pd
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

df = pd.read_pickle('loansData_clean.pickle')

##df = pd.read_pickle('loansData_clean.pickle')

ind_vars = ['Intercept', 'FICO.Score', 'Amount.Funded.By.Investors']

def logistic_function(interest_rate, fico_score, loan_amount):
    df['IR_TF'] = df['Interest.Rate'].apply(lambda x : 0 if (x <= interest_rate) else 1 )
    df['Intercept'] = float(1.0)

    logit = sm.Logit(df['IR_TF'], df[ind_vars])
    result = logit.fit()
    coeff = result.params
    
    linear = float(coeff['Intercept'] + coeff['FICO.Score']*fico_score + coeff['Amount.Funded.By.Investors']*loan_amount)
    p = 1/(1 + math.exp(linear))
    return p

ficos = [x for x in df['FICO.Score'].unique()]

p_values = [logistic_function(0.12, x, 10000) for x in ficos]

plt.figure()
plt.scatter(x = ficos, y = p_values)
plt.show()

print('The p-valuefor FICO Score %d is %f' % (750, logistic_function(0.12, 750, 10000)))
print('The p-value for FICO Score %d is %f' % (720, logistic_function(0.12, 720, 10000)))

