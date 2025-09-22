# Proposal 

## Description
Most of the college students use MBTA as commution method. However, because of the frequent delay of MBTA‘ transit vehicles, students usually suffer from it, being late to the class or even missing the final exam. MBTA delays because of multiple factors, and we want to figure it out, studying the long-term pattern of MBTA. We are interested in investigating the weight of different factors that can affect MBTA's arrival time.

## Goal
 
### Primary Goal
Predict the binary result if MBTA will delay or not based factors we consider.

### Research Goal 
Study the relationships between delay and different factors.

## Data Collection
Scraping data from [https://api-v3.mbta.com/](https://api-v3.mbta.com/) and other data wesbisites, collecting data related to weather, destinations, schedules, routes ...

## Modeling Method 
### Logistic Regression   
It's a base line. Logistic regression is to help us determine the binary result. The advantage is it's straightforward , common , and easy to implement. In addition, it doesn't need paramater tunning. The downside is its prediction accuracy may be low.

### Decision Tree
Output the binary result of delaying or not through feature spliting. It is good at telling what's the most key factors, and it can handle non-linear relationship. The disadvantages is it's sensitive to the noise.

### XGBoost
Comprehensively predict the binary result. It's comprehensive and not sensitive to noise, while it requires parameter tunning and it's not time efficient.

### Deep Learning
It's good at dealing with complicated non-linear problems. It predicts very accurately, but it needs sufficient data set to support it.
...

## Visualization
### Box plot
Delay or not with respect to temperature/rainfall/wind speed/visibility. Compare the distribution of predictor variables which MBTA does not delay and that which MBTA delays.

### Heatmap 
Visualize the extent of delay of each day in 24 hours.

### Roc curve
It helps evaluate the performance of models by calculating true positive rate (TPR) and false positive rate(FPR).

## Test plan
80% as training set, 20% as testing data.