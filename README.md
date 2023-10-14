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

## PROGRAM::
NAME:Naveenaa V.R
REG NO:212221220035

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
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```

```
df
```

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/e33ffa0a-2d9b-4663-a1b2-cbcbd1a0da55)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![image](https://github.com/Naveenaa28/ODD2023-Datascience-Ex06/assets/131433133/83ccda64-91b1-4fc8-b902-7b0817d3db42)












 
