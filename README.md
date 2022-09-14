[![Titanic](/titaniclogo.png "Titanic")](# "Titanic")
# Project 1: Titanic Crash Passenger Data
## Table of contents
- Installation
- File Descriptions
- Project Motivation
- How to interact with Project
- Licensing, Authors, Acknowledgements
## Installation
### Libraries
- `mport numpy as np`
- `mport pandas as pd`
- `import matplotlib.pyplot as plt`
- `from sklearn.linear_model import LinearRegression`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.metrics import r2_score, mean_squared_error`
- `import seaborn as sns`
- `get_ipython().run_line_magic('matplotlib', 'inline')`
### Python version
Python Version 3
## File Descriptions
- **Project1.py**- Python code used to do analysis
- **README.md** - this readme file
- **Titanic.csv** - the data set used of all passengers
## Project Motivation
Overall, the titanic incident is fascinating to many people for many reasons. After looking at the summary data I had several questions I wanted to explore to answer my own questions that were either not readily known by me at the time or things/comparissons I've never seen before for the incident.
#### Main Questions
1. What was the survival rate for children vs. adults?
2. What had the largest correlation to survival?
3. Can I predict who would survive based on the data and linear regression
4. What was the survival rate for those that did not make it to a boat?
## How to interact with the project
### Results
1. “Women and Children!” may not have been accurate. The priorities seemed to be strongly on JUST women. There is also a less extreme priority to higher class folks as well.
2. Of those who did not make it to a boat and STILL survived — there was a much higher chance of survival if they were a woman. This confirms some scientific evidence but research is ongoing.
3. Children did have a higher survival rate — but likely because there were far fewer of them overall.
4. Just making it to a boat was the highest correlation and predictor of survival in the regression models.
5. The difference between the 3 regression models shows very little difference from each other — with very high prediction accuracy overall. 88–89% R² value, 3–5% average error, and at most the models were 16–17% off of the true values.
### Methods Tried
- Linear regression for suvival against all independent numeric variables
- Linear regression for suvival against just whether they were in a boat or not
- Logisitc regression for survival against all independent numeric variables
- Correlation coeficcients & heatmaps
- Subsetting data into:
 - children/adults
 - men/women
 - survivors/non-survivors
 - on lifeboat/not on lifeboat
### How I would Improve
- Experimenting with other types of models to find better fit
- Try pairing passenger data of the titanic with other data availbale about those same individuals to find greater nuance
- Determining and accounting for interactions between variables (ex: Class & Fare price)
## Licensing, Authors, Acknowledgements
- **Source of Data** - [Kaggle](https://www.kaggle.com/c/titanic "Kaggle")
- **Author** - David Krol (Me)
- **Aknowledgments** - StackOverflow, Pandas Docs, Scikit Learn Docs, Numpy Docs
## Link to blog post
https://medium.com/@DaveKrol/this-new-titantic-crash-data-analysis-will-make-you-surprised-at-who-survived-634a452d9a06
