## GEOM90006 Spatial Data Analysis Assignment 4

#### Analysing and forecasting demand of trains in Greater Melbourne

Semester 1 2024, University of Melbourne

#### Group Members

- **Name:** Lang (Ron) Chen **Student ID:** 1181506
- **Name:** Yihan (Tommy) Bao **Student ID:** 1174241 

---

We built models to predict for train demands in suburbs currently without a station, using training data of demand at current stations. Models used include Machine Learning models such as Boosting Machines and Random Forests. Impact of factors (Population, weather, non-residential buildings) were analysed in using econometric methods using linear regression with a Graph Neural Network to encode the spatially distributed station demand data.

Please see our full methodology and results in `./presentables/final_notebook.ipynb`. 

---

### 1.1 Project Title
Spatial Analysis of Train Service Demand in Greater Melbourne

### 1.2 Background Information and Literature Review

Metro Train is a vital service in the metropolitan area of Victoria. It provides connectivity across the region and has become one of the most popular travel options. According to the Department of Transport and Planning annual report, Victorian train services carried 173.8 million passengers from 2022-2023, and 90.39\% are in the metropolitan region (Department of Transport and Planning, 2023). The accessibility for metro train services across Victoria is not uniform but varies significantly based on several factors. For example, a previous study in Melbourne based on public transport coverage has shown that areas with high population density have higher public transport accessibility (Alamri, Adhinugraha, Allheeib, & Taniar, 2023). Furthermore, an accessibility measure extending public transport accessibility levels (PTAL) and supply index (SI) is proposed, and it resulted in a similar conclusion (Saghapour, Moridpour, & Thompson, 2016).

While their approach analysed the frequency and availability of public transport service supplies, they did not incorporate the perspective of actual demand. There are two major factors that influence the demand for public transport: structural factors (such as travel time, distance, and level of transport supply) and external factors (including demographics, city-built environments, government policies, etc.) (Polat et al., 2016). It implies that while population demographics and densities affect metro train usage, non-residential facilities are also important as there are more of them being developed in urban areas, such as shopping centres, schools and hospitals.

Many studies already exist for analysing correlation and predictive relationships between multiple factors in a spatial context. Amongst those commonly utilised are Geographically Weighted Regression (GWR) and Similarity GWR (SGWR). Multiscale GWR (MGWR) extend SGWR by removing its constraints on the local relationship within each model to vary at the same spatial scale (Fotheringham, Oshan, & Li, 2023). Our analysis is inspired by Brent Selby and Kara M. Kockelman, where they compared the performance of kriging and GWR in spatial forecasting (Selby & Kockelman, 2013). We decide to embedd the concept of geographical weights into different regression models, and finally perform kriging for producing a continuous prediction.

### 1.3 Research Question and Topic
Our research question is:
- What is the impact of factors like weather, demographics and non-residential buildings on train demand?
- Which SA2s should be prioritised for new stations?

### 1.4 Motivation and Aim
Both members are frequent travellers by train and are intrigued by varying levels of demand at different stations, and are also curious of the necessity of the current City Tunnel project.

Improving train services and infrastructure based on detailed spatial analysis is crucial for fostering sustainability and resilience in urban areas. From our project, we hope that city planners can optimise resource allocation and enhance public transportation accessibility. Specifically, high-demand areas can be prioritised for capacity expansion and increased service frequency, to reduce growing population's reliance on private vehicles. This approach promotes lower carbon emissions, decreased traffic congestion, and a more sustainable urban transportation network.

Moreover, addressing low-demand areas by improving connectivity and accessibility can enhance the resilience of the transportation network. Ensuring equitable access to reliable train services across all neighborhoods supports social inclusion and economic opportunities, particularly in underserved regions.

To achieve this goal, the purpose of this research is to analyse the factors contributing to the demand for metro trains in Greater Melbourne using spatial data analysis techniques, which builds towards a model to predict demand in locations that currently do not have train stations in an attempt to advise on the location of future constructions. Features that will be analysed include local population density and demographics (age, salary), and the existence of non-residential buildings. The relationship of rainfall to the existing train demands will also be investigated.

### 1.5 Hypothesis

We hypothesise that:
- mean rainfall should have negative effect on demand because travellers are less inclined to use public transport in the rain
- wealth related census attributes to have negative effect on demand because economically better off travellers would prefer private transport
- population to have positive impact on demand as there would be more potential customers
- existance of non-residential buildings to have positive impact on demand as it draws travellers to the region

### 1.6 Scope and Data Sources
We consider following scope to cover our analyses
- Geographical Scope: The boundary of our research is Greater Melbourne since train services beyond this region are sparse and scarce. Within the region, areas are aggregated according to ABS Statistical Area Level 2 
- Temporal Scope: The scope for this project is to predict demand for different spots in Greater Melbourne based on the present. Data used in this project vary in timestamp between 2022 and 2023, but are considered sufficient proxies for the purposes of this project.


**Train Stop location data and Passenger Count data**\
Given the topic's focus, the Victorian train system dataset forms a cornerstone of our analysis. All subsets of this dataset are point data. Source: https://discover.data.vic.gov.au/dataset/train-service-passenger-counts
- Train stop location data gives the geographical location of each station.
- Train passenger count data is recorded from 2022.01 to 2023.07. The data records the number of boarding and disembarking passengers at each station for each service, which could be aggregated by time to be a proxy for the demand for trains at each station.

**Census data**\
This is a polygon (point) from the Australian Bureau of Statistics (https://www.abs.gov.au/census/) containing the median values of various attributes of the residential population in each area.

**Non-residential Facilities Data**\
We decide to use locations of hospitals, sports facilities, schools and shopping centres to estimate the region's train demand. These are places that would contribute to the demand for trains, and cover the population's essential daily activities. The data sources are:
- Shopping Centres: https://www.australia-shoppings.com/malls-centres/victoria
- Hospitals: https://springernature.figshare.com
- Sports Facilities: https://discover.data.vic.gov.au
- Schools: https://www.education.vic.gov.au

**Rainfall Data**\
This is a raster data of Victoria rainfall grid (Bureau of Meteorology) containing average rainfall in victoria region

**Boundary Data**\
They are Polygon/Multipolygon data used to assist our data aggregation (https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files):
- GCCSA Shapefile: Restricts data within our Greater Melbourne scope
- SA2 Shapefile: We decide to conduct analysis in SA2 level


### 2.1 Feature Engineering
<a id="Feature Engineering"></a>

Features are created as follows:

- Rainfall feature: This feature will be the mean annual rainfall in the SA2 which contains the station
    
    - during inference, SA2s will use their own SA2's mean annual rainfall

- Census Total fields (i.e. total population): We hypothesise only the population characteristics in a close proximity will impact the station's demand; so we choose to define the proximity as the circle with **radius = average nearest neighbour distance between points** because this will try to prevent feature values of different stations being too similar (which will make the feature useless). ANN reflects the average spacing between stations in Greater Melbourne, providing an empirical measure based on real-world data. When planning for future infrastructure and demand, having a standardised affecting area allows for more straightforward extrapolation and modelling.
    
    - We then proportionate the polygon point data into the proximity based on intersection of SA2 polygons and the circular proximity. Note that since we are working with totals, the denominator in the fraction of intersection percentage is the SA2 area, as totals are an exogenous variable and how much value it proportionates over depends on how much of the original area is within the new proximity.

    $$
    V_{\text{Station}} = \sum_{\text{SA2s}} \left( \frac{A_{\text{overlap}}}{A_{\text{SA2}}} \times V_{\text{SA2}} \right)
    $$
    
    - during inference, SA2s will assume stations are built at the polygon centre, and a circular proximity will be built from there.

- Census Median fields: We use ANN as radius to build a circular proximity area, with justification exactly the same as above.

    - We then proportionate the polygon point data into the proximity based on intersection of SA2 polygons and the circular proximity. Note that since we are working with medians, the denominator in the fraction of intersection percentage is the proximity area, as medians are an endogenous variable so how much value it proportionates over depends on how much the original SA2 occupies the new proximity.

    $$
    V_{\text{Station}} = \sum_{\text{SA2s}} \left( \frac{A_{\text{overlap}}}{A_{\text{station ANN-radius proximity}}} \times V_{\text{SA2}} \right)
    $$

    - during inference, SA2s will assume stations are built at the polygon centre, and a circular proximity will be built from there.

- Non residential Building features: We use half ANN as radius to build a circular proximity area, using half ANN this time because we don't want the same building to contribute to features for two stations (in real life people would arrive at a specific station to travel to a building). 

    - The feature is then created as a boolean value, taking value 1 if at least one building of the building type for this feature (i.e. School) exists in the proximity, else 0.

    $$
    I_{\text{Building}} = 
    \begin{cases}
    1 & \text{if building(s) exist in } \frac{1}{2} \text{ station ANN-radius proximity} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    - during inference, SA2s will assume stations are built at the polygon centre, and a circular proximity will be built from there.

- Weekday: boolean value which takes on 1 if the day is week day else 0

    $$
    I_{\text{Weekday}} = 
    \begin{cases}
    1 & \text{if weekday} \\
    0 & \text{otherwise}
    \end{cases}
    $$

- Demand for trains at other stations: 
    
    - For Tabular data in Machine Learning: each station's log-demand of the same day will become features for other stations of the same day, with the log-demands weighted down by a Gaussian Kernel depending on distance of stations to the station of interest. To prevent data leak, the weight of own station to own station is set at 0 (no influence).
    
    $$
    w_{ij} = 
    \begin{cases}
    0 & \text{if same station} \\
    \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right) & \text{otherwise}
    \end{cases}
    $$

    where $\sigma^2$ is the variance of all pairwise distance between stations.


    - For Graph Neural Network: features don't need to be explicitly created - each station goes in with their daily log-demand as feature, and a weights matrix containing weights as defined above will efficiently weigh each stations' demand depending on the station of interest within the model.

### 2.2 Machine Learning Features (putting features created above together with their target variable)
<a id="premodelling"></a>

Each training sample point is one daily station's observation. 

The target of the regression is $\log(daily\_totaly\_demand)$ which is the **logged value of the sum of alightings and boardings at that station for the day**.

After features are created, the sample points from the first 70% of days in the training data are made into a training set, the next 15% validation set (for hyperparameter optimisation) and last 15% test (final out of sample). All data are z-scored normalised based on $\mu$ and $\sigma$ of the training set.

#### 2.3 Feature Selection
Feature selection is essential in machine learning for improving model performance and simplifying models.By removing irrelevant or redundant features, it reduces overfitting and enhances accuracy, leading to better predictive performance. Additionally, it simplifies models, making them easier to interpret, which is crucial for urban planners who need to understand the model's decisions. This process ensures that predictions about traffic demand are both efficient and effective, providing clearer insights and more reliable guidance for train development.

In this section, we focus on selecting two types of features, census data (predictor) and train demand data (respond variable). We suspect some features in these categories are correlated with each other.

### 2.4 Models
<a id="models"></a>
We propose 4 different types of models
- Linear Regression: Captures linear relationship between features and responses. It is used as a baseline model to explore how non-linear pattern (such as feature interactions and spatial autocorrelation) exist in the data
- Random Forest: A decision tree based algorithm that is flexible and able to model interactions
- Gradient Boosting (XGB and LGBM): An ensemble learning technique that builds models sequentially, where each new model attempts to correct the errors of the previous ones using gradient descent to minimize a loss function.
- Graph Neural Network: Capturing dependencies between nodes (stations) through message passing and aggregation of node features across graph edges.

### 3.1 Result of ML models
Mean Squared Error (MSE) measures the difference between actual and predictive values, given the formula:

$$
    MSE = \frac{1}{n} \sum^{n}_{i=1}(y_i - \hat{y_i})^2
    $$

Since our response is normalised to range of [0, 1], the possible MSE also lies within this range

On the other hand, R-squared provides a direct interpretation about the proportion of variation in train demand that can be explained by our model, given by:
$$
    R^2 = 1 - \frac{\sum^{n}_{i=1}(y_i - \hat{y_i})^2}{\sum^{n}_{i=1}(y_i - \bar{y_i})^2} 
    $$

### 3.2 Train Demand Prediction
During inference to predict future demand, only 2 instances are created for each station/sa2: one for inferencing mean weekday demand and one for mean weekend demand. The log-mean demand of each station is used to replace the daily demands.

After predictions are made for each station, all predictions are reverse-Z-scored, exponentialised, and the

$$
\text{final\_demand} = \frac{5}{7} \cdot \hat{\text{weekday\_demand}} + \frac{2}{7} \cdot \hat{\text{weekend\_demand}}
$$

### 4.1 Models
From the model performance table, we can see that Graph Neural Network performs the best with 0.9962 $R^2$ out of sample. Light Gradient Boost regressor is the second best performing while Random Forest (which has weaker inductive bias than boosters) is ranked third.

This is likely due to the inherent autocorrelation in geographic data. Although we applied geographically weighted features to all regression models, GNN can encode additional non-linear dependencies through graph representation, further justifying the significance of the spatial property in this task.

### 4.2 Quantitative Feature Analysis


Ordinary Least Squares (Linear Regression) has the ability to provide quantitative explanations of the impact on the dependent variable that changing the independent variables have, through the value of the weight coefficents. It is also able to determine whether each independent variable has statistically significant effects on the dependent variable using t-tests. We approximate this behaviour on the Graph Neural Network by putting a linear layer (linear regression) on top of the Graph Neural Network so the GNN can both capture the non-linear spatial relationships while also producing linear quantitative effects. Below is a table of the coefficients

| Model | Linear Layer Coefficient | GNN Linear Layer Coefficients |  
| --- | --- | --- | 
| Weighted sum of demand of other stations | N/A | 0.3000 |  
| Weekday | -0.039 | 0.0085 |  
| School | 0.700 | 0.0003 |  
| Hospital | 0.410 | 0.0025 |  
| Sport facilities | -0.060 | 0.0020 |  
| Shopping Centres | 0.490 | 0.0121 |  
| Mean Rainfall | -0.570 | -0.0023 |  
| Total Population | 0.486 | 0.0218 |  
| Median weekly rent | -0.410 | -0.0027 |  
| Median weekly mortgage | 0.410 | -0.0074 |  
| Median weekly income |  0.570| -0.0047 |  
| Median weekly household income | -0.770 | 0.0109 |  

Since the target here is log(demand), each 1 unit increase in each variable (standard deviation for continuous variables, or going from no building/not weekend to have building/weekend) will impact demand by increasing $100 \cdot \beta$%. For weighted other stations demand, because the feature itself is also logged, 1 sd movement in weighted other stations demand will cause $\beta$% change in demand for this station.

Through this observation, it is clear that Linear Regression's interpretation is corrupted, as the signs are often in inverse direction compared to our hypothesis, and the magnitude of coefficients are really large (i.e. one wouldn't expect 1 sd increase in average rainfall in melbourne to cause 57% decrease in demand, which is what the Mean Rainfall coefficient of -0.57 is suggesting). While all the linear regression coefficients are still highly statistically significant, it is likely that when applying linear regression to features that have spatial correlation (i.e. stations) it has violated i.i.d. assumptions of the model, which has caused t-stats to be more significant than they should be, while corrupting interpretations. This suggests that while our weighting method for demand can allow Machine Learning models to produce good predictions, it is not useful for interpretation.

On the contray, the 2 layer GNN was able to match our hypothesis. Its analysis of the quantitative impacts of the features are as follows:

- 1% increase in weighted sum of nearby stations has **0.3% increase** in demand for this station
- Weekday has **0.85% more demand** compared to weekend
- Stations with schools nearby has **0.03% demand increase**
- Stations with hospitals nearby has **0.25% demand increase**
- Stations with sports facilities nearby has **0.2% demand increase**
- Stations with shopping centres nearby has **1.21% demand increase**
- 1sd increase in mean rainfall has **0.23% demand decrease**
- 1sd increase in total population has **0.27% demand decrease**
- 1sd increase of meadian nearby weekly mortgage has **0.74% demand decrease**
- 1sd increase of median nearby weekly income has **0.47% demand decrease**
- 1sd increase of median nearby weekly household income has **1.09% demand increase**

*All standard deviation are based on the sample of Greater Melbourne.*

A useful effect of having 2 layer GNN is that secondary effects of circular impacts (i.e. station of interests' demand impacts other stations' demand, which also impacts the station of interests' demand) can be captured, and this is something that the Linear Regression cannot do. This is similar to simultaneous causality models in econometrics.

### 4.3 Sustainability and Resiliance

Understanding the demand for trains based on residential and non-residential factors helps in optimising the allocation of resources. Train services can be adjusted to match the demand, reducing energy consumption and emissions associated with underutilised services. Integrating train demand data with urban planning can ensure that new residential and commercial developments are well-connected to public transport networks. This promotes sustainable urban growth by reducing the need for car travel and encouraging the use of public transportation.

On the other hand, The findings can be used to identify vulnerabilities in the transport network and prioritise investments in infrastructure resilience. This includes reinforcing tracks and ensuring that stations are equipped to handle increased demand during different events.

### 4.4 Limitations and Future Work

While the models that produced the predictions are accurate and confidently matches domain knowledge, several limitations persists:

1. Stations can not be considered for construction independently, rather as part of a train line. Hence, multiple suburbs forming a potential train line should be evaluated together for more rounded analysis. This is something that our model fails to do and a more formal analysis should be conducted in this way using our model and more domain knowledge.

2. Empirical factors sometimes are vital when deciding the location of traffic infrastructures. For example, the elevation of land in potential rail lines should be heavily considered, as it may physically impede the ability to construct rail lines. On the other hand, hydrology and other land uses may also impact empirical design (Song et al., 2021). 

Future works:

1. Incorporate more up to date data and more features into the model for analysis and potentially better predictability. Our current analysis uses census data from 2021 which is serviceable but not precise.

2. Work on GNN's model architecture design to allow its excellent analysis abilities to be transferred to inference.

3. Spatial Temporal forecasting of daily train demand.

### 5. Conclusion
We investigated the quantitative impact that weather, non-residential building and population features had on train station demand, and used predicted demand at each SA2 centre to predict which area should be prioritised for their first station. Quantitative analysis using Graph Neural Network indicated the existance of buildings and greater population to have positive impact on demand, while higher average rainfall and greater median wealth to have negative impacts on train demand. Using Light Gradient Boost Regressor, we forecast Balwyn North, Carlton, Fitzroy, West Melbourne and Hidelberg West to be the top 5 SA2s that should be prioritised for new stations, and this matches domain knowledge (i.e. West Melbourne's booming migrant population, Parkville station being built in Carlton etc). The results of this research can contribute towards more efficient public transport in Greater Melbourne, promoting the cause of sustainability and better liveability.

## References
[1] Alamri, S., Adhinugraha, K., Allheeib, N., & Taniar, D. (2023). Gis analysis of adequate accessibility to public
transportation in metropolitan areas. *ISPRS International Journal of Geo-Information, 12*(5).

[2] Department of Transport and Planning. (2023). *Department of transport and planning annual report 2022-23.*

[3] Fotheringham, A. S., Oshan, T. M., & Li, Z. (2023). *Multiscale geographically weighted regression: Theory and
practice* (1st ed.). CRC Press.

[4] Polat, C., et al. (2016). The demand determinants for urban public transport services: a review of the literature.

[5] Saghapour, T., Moridpour, S., & Thompson, R. G. (2016). Public transport accessibility in metropolitan areas:
A new approach incorporating population density. *Journal of Transport Geography, 54,* 273-285.

[6] Selby, B., & Kockelman, K. M. (2013). Spatial prediction of traffic levels in unmeasured locations: applications
of universal kriging and geographically weighted regression. *Journal of Transport Geography*, 29, 24–32.

[7] Song, T., Pu, H., Schonfeld, P., Zhang, H., Li, W., Peng, X., . . . Liu, W. (2021). Gis-based multi-criteria railway
design with spatial environmental considerations. *Applied Geography*, 131, 102449.
