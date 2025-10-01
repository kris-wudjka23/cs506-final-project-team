# Proposal 

## Description
Most college students use the MBTA as a commuting method. However, due to the frequent delays of MBTA's transit vehicles, students often suffer, being late to class or even missing the final exam. MBTA delays are due to multiple factors, and we want to figure it out, studying the long-term pattern of MBTA. We are interested in investigating the weight of different factors that can affect the MBTA's arrival time.

## Goal
 
### Primary Goal
Predict the binary result if MBTA will delay or not based factors we consider.

### Research Goal 
Study the relationships between delay and different factors.

## Data Collection
Scraping arrival/departure data from [https://api-v3.mbta.com/](https://api-v3.mbta.com/), [https://mbta-massdot.opendata.arcgis.com/](https://mbta-massdot.opendata.arcgis.com/), and other data wesbisites for collecting weather,such as [https://dev.meteostat.net/bulk/normals.html#endpoint](https://dev.meteostat.net/bulk/normals.html#endpoint), destinations, schedules, routes data which may potentially affect the arrival/departure time.

## Modeling Method 
### Logistic Regression   
It's a baseline. Logistic regression is to help us determine the binary result. The advantage is that it's straightforward, common, and easy to implement. In addition, it doesn't need parameter tunning. The downside is that its prediction accuracy may be low.

### Decision Tree
Output the binary result of delaying or not through feature splitting. It is good at telling what's the key factors are, and it can handle non-linear relationships. The disadvantages are that it's sensitive to noise.

### XGBoost
Comprehensively predict the binary result. It's comprehensive and not sensitive to noise, while it requires parameter tuning and it's not time efficient.

### Deep Learning
It's good at dealing with complicated non-linear problems. It predicts very accurately, but it needs a sufficient dataset to support it.
...

## Visualization
### Box plot
Delay or not with respect to temperature/rainfall/wind speed/visibility. Compare the distribution of predictor variables that MBTA does not delay and that which MBTA delays.

### Heatmap 
Visualize the extent of delay of each day in 24 hours.

### Roc curve
It helps evaluate the performance of models by calculating the true positive rate (TPR) and the false positive rate(FPR).

## Test plan
80% as the training set, 20% as the testing data.
