import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

telecom = pd.read_csv("telecom_customer_churn.csv")

# Data Cleaning
print("Initial shape:", telecom.shape)
print("\nBasic Info:")
print(telecom.info())

# Drop duplicates
telecom.drop_duplicates(inplace=True)

# Handling Errors
telecom['Total Charges'] = pd.to_numeric(telecom['Total Charges'], errors='coerce')

# Handling the missing values by adding median at that place
numeric_cols = telecom.select_dtypes(include=['float64', 'int64']).columns
telecom[numeric_cols] = telecom[numeric_cols].fillna(telecom[numeric_cols].median())

# Fill missing categorical values with mode
categorical_cols = telecom.select_dtypes(include=['object']).columns
for col in categorical_cols:
    telecom[col] = telecom[col].fillna(telecom[col].mode()[0])

# Exploratory Data Analysis (EDA)
sns.set(style="whitegrid")

# Customer Status Distribution(Pie Chart)
plt.figure(figsize=(6,6))
telecom['Customer Status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral', 'yellowgreen'])
plt.title('Customer Status Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()


# Churn by Contract Type(Count Plot)
plt.figure(figsize=(8,5))
sns.countplot(data=telecom, x='Contract', hue='Customer Status',palette='Set2')
plt.title("Churn by Contract Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Charges Distribution(Histogram)
plt.figure(figsize=(10,5))
sns.histplot(data=telecom, x='Monthly Charge', hue='Customer Status', kde=True, palette='coolwarm')
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()

# Internet Type vs Churn(Count Plot)
plt.figure(figsize=(8,5))
sns.countplot(data=telecom, x='Internet Type', hue='Customer Status',palette='dark:skyblue')
plt.title("Internet Type vs Churn")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Services Availed by Customers(Count Plot)
services = ['Online Security', 'Online Backup', 'Device Protection Plan', 'Streaming TV', 'Streaming Movies']
plt.figure(figsize=(12,8))
for i, service in enumerate(services):
    plt.subplot(2,3,i+1)
    sns.countplot(data=telecom, x=service, hue='Customer Status', palette=['#FF6F61', '#6B5B95', '#6B8E23'])
    plt.title(service)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tenure vs Churn (Boxplot)
plt.figure(figsize=(10,5))
sns.boxplot(data=telecom, x='Customer Status', y='Tenure in Months')
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.show()

# Payment Method vs Churn
plt.figure(figsize=(10,5))
sns.countplot(data=telecom, x='Payment Method', hue='Customer Status')
plt.title("Payment Method vs Churn")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14, 10))  
corr_matrix = telecom.corr(numeric_only=True)
sns.heatmap(
    corr_matrix, 
    cmap='viridis', 
    annot=True, 
    fmt=".2f",             
    annot_kws={"size": 10},
    linewidths=0.5
)

plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)               
plt.tight_layout()
plt.show()

# Pair Plot
selected_cols = ['Tenure in Months', 'Monthly Charge', 'Total Charges', 'Customer Status']
sns.pairplot(telecom[selected_cols], hue='Customer Status', diag_kind='kde', corner=True)
plt.show()
