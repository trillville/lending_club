# Lending Club Exploratory Data Analysis

I focused my EDA on the approved loan dataset. While there is no doubt plenty of scope for interesting analysis on the rejected loans, perhaps in the form of a model exploring the features most correlated with approval odds, I decided that a deep dive into one dataset would be more interesting than a superficial look at both. 

Since the dataset covers the earliest days of LendingClub all the way up through 2016, I was particularly interested in investigating the evolution of the product and borrower behavior over that time period. I began by visualizing the aggregate quantity of funded loans. Note the exponential growth tapering off in 2015 and beyond. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot1.png)

Given the scope of these data, I was immediately curious how other metrics of borrower and investor behavior changed over time. To begin, I visualized the distribution of borrower specified loan purposes over time. One thing to note is the increasing homogeneity in stated loan purpose up through 2013. Notably, education loans seem to have been completely phased out. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot2.png)

Relatedly, I looked at the evolution of interest rates across borrower stater purposes, loan grade, and time. First, I briefly examined the distribution of interest rates within each purpose. I held time constant for this visualization to avoid showing misleading results that reflected time trends as opposed to within-grade distributions.

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot3a.png)

I also briefly examined the distribution of interest rates within each grade. I also held time constant for this visualization.

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot4.png)

I also sliced the above analysis by grade - perhaps most notably is the significantly outsized proportion of small business loans in the lowest grade (G) bucket. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/Plot3b.png)

Finally, I built a correlation matrix, which shows time trends between interest rates of loans used for different purposes. Credit card loans had the weakest correlations with other loan types, perhaps reflecting changes in credit card rates that did not spill over into other types of debt. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot3c.png)

Given the geographic features contained in this dataset, I decided it would be interesting to explore the relationship between loan interest rate and geographic location. Unfortunately (and understandably!), only the first 3 digits of each zip code was included, so is isn't possible to do zip code or county level analysis. It was suprising to see as much dispersion in the interest rates across states as I did. was a little surprised by the extent of the dispersion of interest rates across states - the average interest rate in D.C. was more than a full percentage point lower than the average rate in Iowa. A good extension of this analysis would be to determine to what extent other correlates with geographic location, such as income, home ownership, and DTI, explain these spreads. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot5.png)

# Lending Club Interest Rate Model

Next, I thought it would be a fun exercise to see the success to which I could re-estimate the LendingClub assigned ‘grades’. I decided to treat it as a regression problem, as opposed to a multi-category classification, because for a given point in time there is essentially zero overlap between the interest rates assigned to different grades. Moreover, the problem of assigning a risk weighted score translates more intuitively to a regression setting, and it's likely that the LendingClub 'grades' are actually derived from the corresponding risk-weighted interest rate (not the other way around!). I used the Boruta algorithm and some feature engineering to put together a useful feature space, and then compared performance across a suite of different models. 

I used these models to get some useful insights into the underlying data set. Variable importance plots and PCA give a sense for the most influential features and the dimensionality of the feature space, respectively.

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/var_imp_plot.png)

The PCA plot reveals the relatively high dimensionality in the dataset. While there is a noticeable "kink" after about 10 principal components, they only cumulatively explian 60% of the variance. 

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/plot6.png)

Due to computing power and time constraints, I only did 5-fold cross validation and trained based on a very small subset (~10,000 rows) of the dataset. Nevertheless, RandomForest and a modestly (tried 50 different combinations of tuning parameters) xgboost model performed quite well. Lasso/PLS both peformed quite poorly, which is not terribly surprising, given the high dimensionality of the udnerlying data. The plots below compare the RMSE and R^2 of each model through each fold of 5-fold cross validation.

NOTE - as the charts below show, one fold of the cross validation gave very very poor results for the linear regression based models. I am going to re-run the k-fold CV tonight with slightly more data to hopefully get cleaner plots!

![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/model_perf_1.png)
![Alt text](https://github.com/trillville/lending_club/blob/master/screenshots/model_perf2.png)




Finally, I noticed that many of the older loans included a description feature, where borrowers would describe there lending situation in greater detail and provide updates to lenders. I analyzed this in a few interesting ways. The word cloud below shows the most frequently used expressions in these borrower descriptions. To get a sense for the different kinds of topics that borrowers discussed, I used latent Dirichlet allocation to cluster the descriptions into 10 categories. The most frequently used words from each category are shown below. <currently loading word cloud>

The fact that this feature was discontinued strongly suggests that the analysts and LendingClub determined that it was not useful in their credit scoring algorithm. Nevertheless, I am curious if one could have any success predicting borrower defaults based on the language they used in their descriptions. A simple way of exploring this would be a logistic regression predicting default based on “word score”. This score would be constructed by iterating through the list of words, and assigning it a likelihood based on the ratio of times that word was associated with non-defaulting vs defaulting loans. In other words, each word would be associated with the likelihood of that word being used in the description of a defaulting loan. Then, one could simply sum across each log likelihood for each word in a given user's description to get the final statistic (defined above as "word score").

