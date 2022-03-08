# Project 2
---
# Sale Price prediction Model:

## Context
Zillow uses a proprietary system for their calculations which is being widely manipulated by listing agents and homeowners to exaggerate the value of most homes on Zillow. This happens because Zillow allows homeowners and listing agents to enter unverified information about homes.

## Problem statement
Build a model that predicts house sale prices dismissing agent manipulated listings that might cause unpredictability in forecasting.

---
## Description of data
1. [`train.csv`](./datasets/train.csv):
This data set contains information from the Ames Assessor’s Office used in computing assessed values for individual residential properties sold in Ames, IA from 2006 to 2010. 
Here is a link to the data documentation: [data description](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt). It has 2051 rows and 81 columns.

2. [`train-data-clean-imputed-1.1.csv`](./data/act_GOOD.csv):
This data set is obtained after cleaning and formatting the original data set [`train.csv`](./datasets/train.csv). It has 1967 rows and 67 columns. All variable types and descriptions can be found in the link to the data documentation: [data description](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt). Next, all the variables (columns) of this data set are displayed

| Variable        |
|-----------------|
| id              |
| pid             |
| ms_subclass     |
| lot_area        |
| overall_qual    |
| overall_cond    |
| year_built      |
| year_remod/add  |
| mas_vnr_area    |
| bsmtfin_sf_1    |
| bsmtfin_sf_2    |
| bsmt_unf_sf     |
| total_bsmt_sf   |
| 1st_flr_sf      |
| 2nd_flr_sf      |
| low_qual_fin_sf |
| gr_liv_area     |
| bsmt_full_bath  |
| bsmt_half_bath  |
| full_bath       |
| half_bath       |
| bedroom_abvgr   |
| kitchen_abvgr   |
| totrms_abvgrd   |
| fireplaces      |
| garage_cars     |
| garage_area     |
| wood_deck_sf    |
| open_porch_sf   |
| enclosed_porch  |
| 3ssn_porch      |
| screen_porch    |
| pool_area       |
| misc_val        |
| mo_sold         |
| yr_sold         |
| saleprice       |
| ms_zoning       |
| lot_shape       |
| land_contour    |
| lot_config      |
| land_slope      |
| neighborhood    |
| condition_1     |
| condition_2     |
| bldg_type       |
| house_style     |
| roof_style      |
| roof_matl       |
| exterior_1sr    |
| exterior_2nd    |
| mas_vnr_type    |
| exter_qual      |
| exter_cond      |
| foundation      |
| bsmt_qual       |
| bsmt_cond       |
| bsmt_exposure   |
| bsmtin_type_1   |
| bsmtin_type_2   |
| heating         |
| heating_qc      |
| electrical      |
| kitchen_qual    |
| functional      |
| paved_drive     |
| sale_type       |

---
## Data Analysis

1. **Data cleaning:**
These are the steps taken to go from the original data set [`train.csv`](./datasets/train.csv) to the cleaned data set [`train-data-clean-imputed-1.1.csv`](./data/act_GOOD.csv), which is afterwards used to proceed with the model development:

    - Formatted all variable names (column names)
    - Checked for missing values
    - Removed all columns that contained more than 5% of null cells
    - Created a dataframe with only the numerical variables
    - Imputed missing values for the rest of columns containing null cells (of the numerical variables)
    - Created a dataframe with only categorical variables
    - Merged the imputed missing values dataframe with the prior created categorical dataframe
    - Removed all rows containing null cells
    - Checked for potential corrupting charachters in the categorical variables
    - Studied correlation between target feature and categoricak variables via One-way ANOVA model
    - Removed variable with no correlation to target feature

2. **EDA and visualization:**
In this section, the data will be analyzed for an overview study of the data set.

First, I checked for obvious outliers based of data documentation recommendation. The next visualizations are "sale price" and "above ground living area" boxplots.

![This is an image](./images/Sale-price-boxplot.png)

As we can see, there are quite a few values that could be considered outliers. Because there are quite a lot and aren't that unique, I continued exploring.

![This is an image](./images/Ab-gr-liv-area-boxplot.png)

To clearly visualize which are the obvious outliers regarding this two variables, I proceeded to plot them together.

![This is an image](./images/Plot-sale_price-ab_gr_area.png)

As shown above, there are two clear outliers describing houses with more than 4,000 sqft which I removed.

Next, I looked into the target feature distribution:

![This is an image](./images/Sale-price-distribution.png)

As we can observe, the distribution is slightly right skewed, but fairly normal. We can expect normality in our predictive sale prices later on. 

Lets move on to some correlation scores between "sale price" and all the numerical variables.

![This is an image](./images/Corr-heatmap.png)

As we can observe, the numerical variables most correlated with the the target variable are:

    - overall_qual
    - total_bsmt_sf
    - 1st_flr_sf
    - gr_liv_area
    - garage_cars
    - garage_area
    
For the categorical variables, I used a One-way ANOVA test. After the test, it seemed there is no significant information due to the fact that all p-values are practically zero.


## Model development 1

Once finished with the data cleaning and EDA, the resulting predictive features chosen for the model are: 

    - overall_qual
    - total_bsmt_sf
    - gr_liv_area
    - garage_area
    - neighborhood,
    - exter_qual,
    - foundation,
    - bsmt_qual
    - kitchen_qual
    
These features were not only chosen for their strong correlation with "saleprice" but there is research that shows square-footage, bath and kitchen upgrades, good maintenance, neighborhood, and others are all key factors on house valuations.

1. **Loading data:**
First, we import the needed libraries and load the cleaned data we prepared earlier.

2. **Matrixes creation:**
Predictive feature matrix and target matrix are created. Because of categorical variables, I had to dummify these columns.

3. **MLR model:**
In this step, I instantiated a linear regression model. I proceeded with a cross validation, which allows you to see how the model performs on unseen data several times. I also got the R-squared metric along with some residual regression metrics like the mean squared errors (MSE).
After getting these scores, a slight high variance can be detected due to having a higher R-squared score on the training data than the testing one. Also, the training MSE is smaller.

Bellow, I attached a dataframe that dislays features and their coefficients.

![This is an image](./images/Coef-1.png), ![This is an image](./images/Coef_2.png)

These coefficients indicate (choosing numerical and categorical variable randomly as example):

    - Holding all else constant, for every one unit increase in "overall_qual", we expect "saleprice" to increase by $1.101709e+04.
    
    - Holding all else constant, for every one unit increase in "kitchen_qual_Fa", we expect "saleprice" to decrease by $4.585501e+04 relative to "kitchen_qual_EX".
    
    
The next step, is to verify LINE assumptions:

    - **Linearity:**  𝑌  must have an approximately linear relationship with each  𝑋  variable.
    - **Independence of Errors:** Errors (residuals)  𝜀𝑖  and  𝜀𝑗  must be independent of one another for any  𝑖≠𝑗 .
    - **Normality:** The errors (residuals) follow a Normal distribution with mean 0.
    - **Equality of Variances:** The errors (residuals) should have a roughly consistent pattern, regardless of the value of the  𝑋  variables. (There should be no discernable relationship between the  𝑋  variable and the residuals.)
    - **Independence of Predictors (almost always violated at least a little!):** The independent variables  𝑋𝑖  and  𝑋𝑗  must be independent of one another for any  𝑖≠𝑗 .
   

Linearity:

![This is an image](./images/Linearity-scatter.png)

Normality:

![This is an image](../images/Normality.png)

Equality of variances:

![This is an image](./images/Equality-variances.png)

Independence of predictors:

![This is an image](./images/Independence-pred.png)

4. **Ridge model:**
To try to fix the slight high variance, I proceeded to develope a Ridge and Lasso model.

First, I looked for the optimal alpha parameter and then obtained the R-squared and MSE scores.

5. **Lasso model:**
Finally, I develope a Lasso model to compare it's performance to the prior models.

6. **Model scores summary:**

These are the training and testing R-squared scores for the three different models developed.

| Model | MLR        | Ridge      | Lasso      |
|-------|------------|------------|------------|
| Train | 0.89014536 | 0.89001189 | 0.89014531 |
| Test  | 0.86543344 | 0.86551810 | 0.86543543 |


Prioritizing best adaptation to new unseen data, the Ridge model is the one that best fits the problem statement.


## Model development 2

This time, I did the exact same steps as before, but transforming the target variable "saleprice" into a logarithmic scale. This, helped with the linearity between predictive features and itself and improved it's normality distribution.

Linearity:

![This is an image](./images/Linearity-2.png)

Normality:

![This is an image](./images/Normality-2.png)

Equality of variances:

![This is an image](./images/Equality-2.png)


Finally, here are the R-squared results of the three models, transforming the target variable "saleprice" to a logarithmic scale.

| Model | MLR        | Ridge      | Lasso      |
|-------|------------|------------|------------|
| Train | 0.87265    | 0.87240    | 0.87210    |
| Test  | 0.86763    | 0.86883    | 0.86859    |

In this case, the R-squared scores decreased a little, but the distance between training and testing R-squared scores has been significantly reduced. Prioritizing best adaptation to new unseen data, the Lasso model is the one that best fits the problem statement.

---
## Conclusion and recommendations

Because the goal of this project is build a model to predict home sale prices as accurately as possible, dismissing possible listing agent and homeowners manipulation, it's very important to do very good feature selection. 
Transforming the "saleprice" target variable to a logarithmic scale, a Lasso model suits te purpose best because it's the best one at generalizing to new unseen data.
I would recommend to keep optimizing the model and monitoring performance. Last but not least, it is crucial for Zillow to verify all data inputs.