import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\attration analysis\Attration.csv")
df.info()
df.describe()
df.isnull().sum()



#Histgram for age 
plt.hist(df['Age'],bins=20)
plt.title('Age Distribution')
plt.show()

#Bar plot for attrition
sns.countplot(x='Attrition',data = df)
plt.title('Attrition Count')
plt.show()

#Boxplot comparing Monthy income by Attrition

sns.boxplot(x='Attrition',y='MonthlyIncome',data=df)
plt.title('Monthly Income by Attrition')
plt.show()


#correlation heatmap

plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(),annot=True,fmt=".2f",cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# Droping columns which don't need

df.drop(['EmployeeCount','StandardHours','Over18'], axis=1, inplace=True)

# Identify categorical columns except the target (Attrition)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Attrition')  # Since you already converted Attrition separately

# Convert all categorical columns to dummy variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Convert Attrition to numeric
df = pd.get_dummies(df, columns=['Attrition'], drop_first=True)

# Separate features and target
X = df.drop('Attrition_Yes', axis=1)
y = df['Attrition_Yes']

# Then split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
#scaling feature

X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the model
model = LogisticRegression(max_iter=1000,class_weight='balanced')




# Train scaled the model


model.fit(X_trained_scaled,y_train)


# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# coefficients and features

coefficients = model.coef_[0]
feature_names = X.columns

coef_df = pd.DataFrame({'Feature':feature_names,'Coefficient':coefficients})

coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()
coeff_df_sorted = coef_df.sort_values(by='AbsCoefficient',ascending=False)

print(coeff_df_sorted[['Feature','Coefficient']])

print(y.value_counts(normalize=True))


#Decision tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
y_pred_dt =dt.predict(X_test)

# Random forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

#evaluate all models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))            

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))            

from imblearn.over_sampling import SMOTE

# Assuming X_train_scaled and y_train are your scaled features and labels for training

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_sm.value_counts())
