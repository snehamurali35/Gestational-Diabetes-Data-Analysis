---

Logistic Regression for Diagnosis of Gestational Diabetes using Binary Classification Methods
Overview:
Gestational Diabetes (GD) is not only the most prevalent disease among pregnant women, affecting 15 to 25 % of pregnancies around the world, but also exposes women to the potential of developing Type 2 Diabetes later in their lives. While most research revolves around Type 2 Diabetes and its causes, I wanted to illuminate the dominant causes of GD to aid early prevention efforts.
Data:
The dataset utilized contains 7 columns, with 6 independent variables, and 1 dependent variable.
Independent: Age, Pregnancy Number, Weight, Height, BMI, Heredity(binary) → 0 indicates that it doesn't run in the family, 1 that it does.
Dependent: Prediction (binary) → 0 indicates no GD, 1 indicates GD

---

Process + Model:
I utilized a logistic regression model, along with the sklearn and pandas libraries to clean, organize, and analyze the dataset.
Outliers → To handle outlier values in the dataset, that had the potential to skew the prediction results, I computed the Z-score for 'Weight', 'Height' and 'BMI' 
and replaced values in these columns that fell outside the threshold of 3 standard deviations from the mean.
Here, imputation is performed to replace outliers. Values beyond three SD from the mean are replaced by the mean of the column.Then, the independent and dependent 
variables are defined as specified above. With the Y (dependent) variable being the positive of negative diagnosis of GD.Next, the dataset is split into the testing and 
training set, with a test size of 20% and a training set of 80% of the data with a random state of 20, to ensure identical training and testing sets across varying executions. 
The logistic regression model was fit to the x training data, and the y training data in order to predict the presence or absence of GD.
The prediction values are calculated based on the x - testing values (IV) generated from the train test split.


from sklearn.model_selection import train_test_split
# Split into testing and training set 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 20)
print(X_test)

from sklearn.linear_model import LogisticRegression
# create an instance of the model
log_model = LogisticRegression(C = 10.0)
# fit it on the data 
log_model.fit(X_train, y_train)
# now, test the model on the data 
y_pred = log_model.predict(X_test)

---

Results:
An accuracy value is valuable metric to assess logistic regression models, and it is simply the total number of correct predictions / the total number of data points. 
The accuracy calculated was 86.67 %.
Confusion Matrix
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()
The confusion matrix generated aids in organizing the results of the model's prediction by displaying the number of true positives, false positives(1st column), 
and false negatives, true negatives (second column). 153 true positives (1's) were calculated, while 23 true negatives were calculated.

---

Other measurements of accuracy utilized are ROC Curve (Receiver Operating Characteristic Curve) and AUC (Area Under Curve)
ROC curves calculate the true positive and false positive rates, and thresholds, and inform how the model performs compared to chance(randomness).
AUC shows the probability that the model will accurately calculate positive diagnoses, and negative diagnoses (1 and 0 respectively).
A high AUC score means that the model can more accurately discriminate between patients with GD and those without.
y_pred_probs = log_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color = 'purple', linewidth = 5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, y_pred_probs))
AUC Score: 0.86257 → 86.26 %
The dotted line represents 0.5, chance, where a model would randomly guess a diagnosis. A score of 86.26 % indicates that the model has a relatively high ability to 
distinguish between positive and negative classes (with and without GD).

---

Influence of Each Variable

weights = pd.Series(log_model.coef_[0], index = X.columns.values)
# This code shows the coefficient of each IV on the prediction.
Heredity has a coefficient of 2.087, indicating that it is the dominant factor influencing gestational diabetes.

Conclusion:
Pregnant women or those looking to have children should test whether diabetes runs in the family and whether they are predisposed to it in order to take early
prevention efforts like changing their diet, and/or exercise regimen to reduce their chances of getting diagnosed with GD. While BMI has slight impact on GD, 
data analysis indicates that pregnancy number does not; as a result, heredity and BMI should be primarily considered by health professionals working to reduce GD diagnoses.
Future Steps:
Some futher steps involve finding a larger dataset, with more independent variables and 1,000 + rows to increase the validity of results. 
Also, I wish to apply a cluster model or trees model to increase accuracy of diagnosis.
