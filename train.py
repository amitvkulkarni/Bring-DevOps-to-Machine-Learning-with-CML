import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df_heart = pd.read_csv('SAHeart.csv', index_col=0)
df_heart.head()
df_heart.describe()
#df_heart.drop('famhist', axis=1, inplace=True)
df_heart = pd.get_dummies(df_heart, columns = ['famhist'], drop_first=True)

# Set random seed
seed = 52
# Split into train and test sections
y = df_heart.pop('chd')
X_train, X_test, y_train, y_test = train_test_split(df_heart, y, test_size=0.25, random_state=seed)

# Build logistic regression model
model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train, y_train)

# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)


# Confusion Matrix and plot
cm = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.tight_layout()
plt.savefig("cm.png",dpi=120) 
plt.close()

# Print classification report
print(classification_report(y_test, model.predict(X_test)))

#roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Plot the ROC curve
model_ROC = plot_roc_curve(model, X_test, y_test)
plt.tight_layout()
plt.savefig("roc.png",dpi=120) 
plt.close()
