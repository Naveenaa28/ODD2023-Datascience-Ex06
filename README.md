# ODD2023-Datascience-Ex06
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.
## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
STEP 1:Read the given Data

STEP 2:Clean the Data Set using Data Cleaning Process

STEP 3:Apply Feature Transformation techniques to all the features of the data set

STEP 4:Print the transformed features

## DEVELOPED BY:
``
NAME:Naveenaa V.R
REG NO:212221220035
```

```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/Data_to_Transform.csv")
```
```
df
```

![1](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/3cf66d66-8efb-4e5b-a0ed-6ffa4bd3ba96)

```
np.log(df["Highly Positive Skew"])
```
![2](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/f11db8e0-240c-4ebd-a8c1-00992cc6a039)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/110c2ffc-fe30-475a-bec1-dc155f1f2b24)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot (72)](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/877a5b9b-4617-4f7e-a474-9514602ef07b)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/8875e41b-8c5e-4f49-b602-715f8ec0c49e)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
```
```
df
```
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/298b2fd9-8182-4c7f-ad56-74e988d07a37)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
```
from sklearn.preprocessing import QuantileTransformer
```
```
qt=QuantileTransformer(output_distribution='normal')
```
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```

```
df
```

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/e33ffa0a-2d9b-4663-a1b2-cbcbd1a0da55)
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/d4e9f315-28f7-46f5-9b11-7e8a24e62c70)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/35606014-aee2-4169-a518-7a4b9c3a684c)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/fda6e83f-0a06-422f-91cd-e3b80f0aec97)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/9930abc0-3fae-4121-a02a-9c6511cdd2f9)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/94c42ac8-54c2-4c91-8610-afc59304d0e4)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/a4e99713-1b43-40ca-a1de-b8d241a1886b)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/23051a87-a23f-4c69-91c1-60ce51dee103)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/ac326a53-b6e3-4588-a13c-2b0bffe352b2)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/1468ae50-884b-499a-9e05-6e55f1b1ddee)

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/d4f725dd-1c93-4784-8533-4c88a79d6e19)
## RESULT:
Thus feature transformation is done for the given dataset.























 
