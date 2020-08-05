# Predicting_GDP per capita

## **Goal of Project**

The goal of this project is to understand the dataset, get some insights about it and finally train a model to predict GDP ($ per capita).

![download (3)](https://user-images.githubusercontent.com/42713212/89377966-7088d980-d725-11ea-9835-6dc79286aaa9.png)

## **About the dataset**

We will be using the Countries of the World dataset from kaggle (link:https://www.kaggle.com/fernandol/countries-of-the-world)

20 columns of dataset:
* Country
* Region
* Population
* Area
* Pop. Density (per sq. mi.)
* Coastline (coast/area ratio)
* Net migration
* Infant mortality (per 1000 births)	
* GDP ($ per capita)	
* Literacy (%)	
* Phones (per 1000)	
* Arable (%)	
* Crops (%)	
* Other (%)	
* Climate	
* Birthrate	
* Deathrate	
* Agriculture	
* Industry	
* Service

## **Interesting EDA on dataset**

The below join plot graph not only shows the linear relationship between GDP per Capita, but it also shows the concentrations that countries belong in on the graph. There is a very high number of countries that have a very long GDP per Capita and also a very long literacy rate. This highlights the importance of literacy rates in increasing the GDP per Capita of a country.

![download (2)](https://user-images.githubusercontent.com/42713212/89377056-9f05b500-d723-11ea-854e-575ea9e40c8e.png)

Very cool graph that is plotted using plotly. Plotly provides an interactive graph showing the GDP per Capita of the country when the cursor hovers over it. It also shows the possible standard deviation of the different regions, whether the standard deviation is large or small. The dataset has 11 possible regions: 
- ASIA (EX. NEAR EAST)
- BALTICS
- C.W. OF IND. STATES
- EASTERN EUROPE
- LATIN AMER. & CARIB
- NEAR EAST
- NORTHERN AFRICA
- NORTHERN AMERICA
- OCEANIA
- SUB-SAHARAN AFRICA
- WESTERN EUROPE

For example, for region such as NORTHERN AFRICA, it is clear from the plotly graph that the region will have a very low median GDP per capita and that the standard deviation will be small. Similarly, for the region of sub-suharan africa, the region will also have a small standard deviation and an even lower GDP per capita, judging from the whiteness of the plotly graph, which indicates a very low GDP per capita. 

In contrast, the region of ASIA (EX. NEAR EAST) has a very large standard devation due to the presence of countries that have very high GDP per capita and very low GDP per capita. For example Cambodia(1900), China(5000), South Korea(17800) and Japan(28200) are located in the region. These countries have very different GDP per capita. The same goes to the region of Northern America. Even though high GDP per capita countries such as United States(37800) and Canada(29800) lies in that region, low GDP per capita countries such as Mexico(9000) also belongs there. Therefore, the very different colours on the graph of countries in both of the regions stated shows a very high standard deviation. 

![newplot](https://user-images.githubusercontent.com/42713212/89377910-564efb80-d725-11ea-9abb-441f6109d16f.png)

I also plotted a boxplot diagram to confirm whether the above observations based on the plotly graph is correctly. The above observations are valified again by the boxplot diagram which shows the median, standard deviation and outliers of the regions clearly.

![download (4)](https://user-images.githubusercontent.com/42713212/89397832-ebacb880-d742-11ea-8f5d-4a1b6bcac7bc.png)

## **Brief summary of data pre-processing**

**Handling missing data**

Showing number of null values as a percentage. From the data, we can see that 14 out of 20 columns have null values. The column with the largest null value is climate, with a null percentage of 9.69. However, the null percentage is still quite small (<10%).

![download (9)](https://user-images.githubusercontent.com/42713212/89403172-7fce4e00-d74a-11ea-85a8-449d0b0959f4.png)

Interesting heatmap that not shows the density and rows that have missing data.

![download (10)](https://user-images.githubusercontent.com/42713212/89403154-7b099a00-d74a-11ea-90cd-4f3d87875632.png)

```python
data['net_migration'].fillna(0, inplace=True)
data['infant_mortality'].fillna(0, inplace=True)
data['gdp_per_capita'].fillna(2500, inplace=True)
data['literacy'].fillna(data.groupby('region')['literacy'].transform('median'), inplace= True)
data['phones'].fillna(data.groupby('region')['phones'].transform('median'), inplace= True)
data['arable'].fillna(0, inplace=True)
data['crops'].fillna(0, inplace=True)
data['other'].fillna(0, inplace=True)
data['climate'].fillna(0, inplace=True)
data['birthrate'].fillna(data.groupby('region')['birthrate'].transform('mean'), inplace= True)
data['deathrate'].fillna(data.groupby('region')['deathrate'].transform('median'), inplace= True)

# For monaco, i will set the value for industry and service to be 0.05 and 0.78 respectively 
data['industry'][138] = 0.05
data['service'][138] = 0.78
print(data['industry'][138])
print(data['service'][138])

# For western sahara, i will set the value for agriculture and industry to be 0.35 and 0.25 respectively.
data['industry'][223] = 0.25
data['agriculture'][223] = 0.35
print(data['industry'][223])
print(data['agriculture'][223])

data['agriculture'].fillna(0.15, inplace=True)
data['service'].fillna(0.8, inplace=True)
data['industry'].fillna(0.05, inplace= True)
```
**Scaling**

I will be using StandardScalar to standardise the features. It does so by removing the mean and scaling to unit variance.

z = (x - u) / s

```python
sc_X = StandardScaler()
X2_train = sc_X.fit_transform(X_train)
X2_test = sc_X.fit_transform(X_test)
y2_train = y_train
y2_test = y_test
```
**Feature Selection**

We will select only a portion of our features, the ones with coreelation score larger than -/+ 0.3 with gdp_per_capita.

## **Models that I tried out to train the dataset**

**6 models are used to train the dataset:**
- Linear Regression (base model)
- L1 (Lasso) Regression
- L2 (Ridge) Regression
- SVM
- Random Forest
- Gradient Boosting

**Optimization for Random Forest and Gradient Boosting**

Random Forest

* max_features = max number of features considered for splitting a node : ['sqrt', 'auto'],
* min_samples_leaf = min number of data points allowed in a leaf node : [1, 3, 5]
* max_depth = max number of levels in each decision tree (left it out because it takes too long to run)
* min_samples_split = min number of data points placed in a node before the node is split : [2, 5, 10]
* n-estimators = number of trees in the foreset : [100,500,1000]
* bootstrap = method for sampling data points (with or without replacement) : [False, True]

Gradient Boosting

* n-estimators: 100, 500, 1000
* learning_rate: 0.001, 0.01, 0.1, 1
* max_depth: 3, 5, 8
* subsample = The fraction of observations to be selected for each tree. Selection is done by random sampling: 0.7, 1 (Values lower than 1 generally lead to a reduction of variance and an increase in bias)
* min_samples_leaf: 1, 20
* min_samples_split: 0.5-1% of our data --> we have 227 datapoints --> 10 -20
* max_features: 4, 7 (sqrt of number of features is a good guess)

## **Results**

Feature importance for linear regression model. From the bar graph, it is derived that the importance of phones is almost 3 times more important than the next most important feature which is region_WESTERN EUROPE. The least important feature in X4 is region_ASIA (EX NEAR EAST) which makes sense due to the huge variation in GDP per Capita in the region. For example,Cambodia(1900), China(5000), Singapore(23700) and Japan(28200) are located all in the region with very huge varying levels of income. Therefore it is can be understandable that region_ASIA (EX. NEAR EAST) plays a small importance in determining GDP per Capita/ unable to predict GDP per Capita. It would be good to use these observations into narrowing the features that will be use to determine GDP per Capita.

![download (8)](https://user-images.githubusercontent.com/42713212/89399454-1861cf80-d745-11ea-886e-abb5e41677b7.png)

**Bar Graphs showing which model does best based on following metrics**

Metrics used:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R2_Score)

**MAE accuracy score** (Taking into account that the gdp_per_capita values in the dataset ranges from 500 to 55100 USD)

![download (5)](https://user-images.githubusercontent.com/42713212/89399336-e7819a80-d744-11ea-9784-64c73bd0b3af.png)

**RMSE accuracy score** (Taking into account that the gdp_per_capita values in the dataset ranges from 500 to 55100 USD)

![download (6)](https://user-images.githubusercontent.com/42713212/89399334-e6506d80-d744-11ea-930e-2aca42f98dfb.png)

**R2_Score accuracy score** 

![download (7)](https://user-images.githubusercontent.com/42713212/89399329-e486aa00-d744-11ea-8133-1745e191481e.png)

## **Conclusion**

| Accuracy Indicators | Linear Regression  | Ridge Regression | Lasso Regression | Optimized SVM | Random Forest | Gradiant Boosting |
| --- | --- | --- | --- | --- | --- | --- |
| `MAE` | 2854.65 | 2854.61 | 2852.57 | 6386.99 | 2127.80 | 2093.38 |
| `RMSE` | 3760.87 | 3760.74 | 3784.96 | 9131.48 | 3065.70 | 3124.43 |
| `R2_Score` | 0.83 | 0.83 | 0.83 | -0.0091 | 0.89 | 0.88 |

In this project, we used countries_of_the_world dataset to predict GDP per Capita. We used 6 different learning regressors (Linear Regression, L1 and L2 regularization, SVM, Random Forest, and Gradiant Boosting) were tested.

Depending on the accuracy indicator, Random Forest and Gradient Boosting performed the best while SVM acheived the worst performance of the 4.

The best prediction performance was acheived using 

1) Random Forest regressor, using all features in the dataset, and resulted in the following metrics:

* Mean Absolute Error (MAE): 2127.80
* Root mean squared error (RMSE): 3065.70
* R-squared Score (R2_Score): 0.89

![download (11)](https://user-images.githubusercontent.com/42713212/89403286-a8564800-d74a-11ea-93d6-496d5b79f134.png)

2) Gradient Boosting regressor, using all features in the dataset,

* Mean Absolute Error (MAE): 2093.38
* Root mean squared error (RMSE): 3124.44
* R-squared Score (R2_Score): 0.88

![download (12)](https://user-images.githubusercontent.com/42713212/89403280-a68c8480-d74a-11ea-94fd-98f0857871b9.png)







