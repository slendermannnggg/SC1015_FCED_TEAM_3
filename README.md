# Phishing Detection #

Our project explores solutions to tackle the ever-growing threat to cybersecurity - phishing.

<mark>**Problem definition**: To develop a machine learning model that can reliably detect phishing links</mark>

Our Dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/simaanjali/tes-upload)

---
### Exploratory Data Analysis
---
We began by *removing any na* values via the `.dropna()` command.

We then plotted **box and whisker** diagrams to better *visualise the data*.

To further clean the dataset, we fitted our dataset into a preliminary model, plotting a bar chart that measures the importance of each dependent variable in the model's decision making process (*feature importances*).

`importances = clf.feature_importances_`

Upon analysing the plot, we *removed unnecessary columns* that had little to no impact in helping predict phishing; columns removed include '*Unnamed: 0*', '*AtSymbol*', '*IpAddress*', '*NumPercent*'.
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
We also tested the new and improved *Random Tree model* which uses optimal hyperparameters and displayed its *confusion matrices* for its train and test sets.

Last but not least, we used a [pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) from the imblearn module.

With the help of a *preprocessor*, *SMOTE* and our newly optimised *Random Forest model*, our pipeline produced the best model thus far.

We assessed our final's model's performance using standard evaluation metrics such as *accuracy*, *precision*, *recall*, *F1-score* (area under the Receiver Operating Characteristic Curve).  

This was followed by a final set of *confusion matrices*.

Although our classification accuracy did decrease, our **false negative rates plummetted** too. In fact, the main cause of the decrease in classification accuracy is the substantial increase in false positive values. We believe this is acceptable in the case of fraud detection models, because accidentally warning a user of a link is *better* than to accidentally inform them that the link is safe when it is not.

In conclusion, our model's performance increased greatly.

---
### References
---
- [numpy](numpy.org/doc) 
- [pandas](https://pandas.pydata.org/docs/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [scikitlearn](https://scikit-learn.org/stable/)
- [imblearn](https://imbalanced-learn.org/stable/)
- [decision tree](https://scikit-learn.org/stable/modules/tree.html)
- [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [imb-pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)

---
### Acknowledgements
---
EDA - Bo Yu

Decision Tree - Surya

Grid Search - Surya

Pipeline - Bo Yu

TA - Dr. Nikita Kuzmin













