# Predicting_GDP per capita

**Goal of Project**

The goal of this project is to understand the dataset, get some insights about it and finally train a model to predict GDP ($ per capita).

**About the dataset**

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

**Interesting EDA on dataset**

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


![download (3)](https://user-images.githubusercontent.com/42713212/89377966-7088d980-d725-11ea-9835-6dc79286aaa9.png)

