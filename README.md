# Phishing Detection #

Our project explores solution to tackle the ever-growing threat to cybersecurity - phishing.

Our Dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/simaanjali/tes-upload)

### Pre-requisite modules 
- [numpy](numpy.org/doc) 
- [pandas](https://pandas.pydata.org/docs/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [scikitlearn](https://scikit-learn.org/stable/)
- [imblearn](https://imbalanced-learn.org/stable/)

---
### Exploratory Data Analysis
---
We began by removing any *na* values via the `.dropna()` command.

We then plotted **box and whisker** diagrams to better visualise the data.

To further clean the dataset, we removed unnecessary columns, including '*Unnamed: 0*', '*AtSymbol*', '*IpAddress*', '*NumPercent*'.
```
X = data.drop(columns = ['Phising', 'AtSymbol', 'IpAddress', 'NumPercent'])
Tar = data['Phising']
```


---
### Modelling the Data
---
We initally used a **Decision Tree Classifier**. We have shown its *confusion matrices* (for both test and train sets) and values for its *classification accuracy*, *True Positive rate*, *True Negative rate*, *False Positive rate* and *False negative rate*. 

We then compared it with a **Random Forest Classifier**, also depicting its *confusion matrices*.

As expected, the random forest classifier did not give a significant increase in predicting accuracy.  

---
### Optimisation
---
We optimised the *Random Tree model* by using **grid search** to obtain optimal hyperparameters and **cross-validation** to evaluate our model. Due to time and memory limitations, we had to restrict the values of hyperparameters tested.

```
param_grid =
{
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
```
We also tested the new and improved *Random Tree model* which uses optimal hyperparamters and displayed its *confusion matrices* for its train and test sets.

Last but not least, we used a [pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) from the imblearn module.

With the help of a *preprocessor*, *SMOTE* and our newly optimised *Random Forest model*, our pipeline produced the best model thus far.

We assessed our final's model's performance using standard evaluation metrics such as *accuracy*, *precision*, *recall*, *F1-score* (area under the Receiver Operating Characteristic Curve).  

This was followed by a final set of *confusion matrices*.

In conclusion, our model's performance increased greatly.

---
### Acknowledgements
---
EDA - Bo Yu

Decision Tree - Surya

Grid Search - Surya

Pipeline - Bo Yu

TA - Nikita Kuzmin













