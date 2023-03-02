import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

app = pd.read_csv('./csv/application_record.csv')
credit = pd.read_csv('./csv/credit_record.csv')


# ----// Application Records cleaning //-----


# search for duplicates rows
total_rows = app.shape[0]
unique_id = app['ID'].nunique()
print(total_rows, unique_id)

# it is noticeable that are duplicates rows since the number of unique ids
# is significantly smaller than the number of rows
# therefore a measure of dropping the duplicates should be used

app = app.drop_duplicates('ID', keep='last')
print(app.info())

# searching the data for null values
plt.figure(figsize=(10, 15))
sns.heatmap(app.isnull())
plt.show()

# looking at the heatmap it is noticeable that the OCCUPATION_TYPE has a significant large
# amount of null values there for we will drop this column
app.drop('OCCUPATION_TYPE', axis=1, inplace=True)

# filtering all the categorical features in the dataframe
ot = pd.DataFrame(app.dtypes == 'object').reset_index()
object_type = ot[ot[0] == True]['index']
print(object_type)

for feature in object_type:
    print(feature, "values: ", app[feature].value_counts())

app.CODE_GENDER = app.CODE_GENDER.apply(lambda x: 1 if x == 'M' else 0)
app.FLAG_OWN_CAR = app.FLAG_OWN_CAR.apply(lambda x: 1 if x == 'Y' else 0)
app.FLAG_OWN_REALTY = app.FLAG_OWN_REALTY.apply(lambda x: 1 if x == 'Y' else 0)

NAME_EDUCATION_TYPE_VALS = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4
}
app.NAME_EDUCATION_TYPE = app.NAME_EDUCATION_TYPE.apply(lambda x: NAME_EDUCATION_TYPE_VALS[x])

object_type.drop(object_type[object_type.isin(['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE'])].index)

one_hot_app = pd.get_dummies(app[object_type])
app = pd.concat([app, one_hot_app], axis=1)
app.columns = app.columns.str.replace(' ', '_')

app.drop(columns=object_type, axis=1, inplace=True)
print(app.info())

# ----// Credit Records cleaning //-----

total_rows = credit.shape[0]
unique_id = credit['ID'].nunique()
print(total_rows, unique_id)

# We will repeat the same process for the credit records dataframe

credit = credit.drop_duplicates('ID', keep='last')
print(credit.info())

# searching the data for null values
sns.heatmap(credit.isnull())
plt.show()

credit['MONTHS_BALANCE'] *= -1

print(credit['STATUS'].value_counts())

# in the second dataframe its seems like all values are non-null values and the
# two dataframes are ready for merge

def classify_past_due(value):
    if value in ['X', 'C']:
        return 0
    elif value in ['0','1', '2', '3', '4', '5']:
        return 1
    else:
        return None

# apply the function to the 'past_due' column and create a new 'past_due_class' column
credit['STATUS'] = credit['STATUS'].apply(classify_past_due)

status_vals_dist = credit['STATUS'].value_counts(normalize=True)
status_vals_dist.plot.bar()


# show plot
plt.show()


df = pd.merge(app, credit, on="ID", how="left")
df = df.dropna()

print(df.info())

# Compute the correlation matrix
corr = df.corr()
corr = abs(corr)

# # Mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
#              square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# plt.show()


# Mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True, fmt='.2f')
plt.show()