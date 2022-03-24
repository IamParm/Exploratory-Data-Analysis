# %% Importing libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# %% Loading the dataset and quick look at first 5 rows
penguins_size = pd.read_csv('penguins_size.csv', sep = ",")
penguins_size.head()

# %% dataset shape
print("Shape is: ", penguins_size.shape)

# %% Quick look on all the columns
penguins_size.columns

# %% features information
penguins_size.info()

# %% datatypes of features
penguins_size.dtypes

# %% looking into null values
penguins_size.isnull().sum()

# %% Imputing the missing values with mean
penguins_size["culmen_length_mm"] = penguins_size["culmen_length_mm"].fillna(value = penguins_size["culmen_length_mm"].mean())
penguins_size["culmen_depth_mm"] = penguins_size["culmen_depth_mm"].fillna(value = penguins_size["culmen_depth_mm"].mean())
penguins_size["flipper_length_mm"] = penguins_size["flipper_length_mm"].fillna(value = penguins_size["flipper_length_mm"].mean())
penguins_size["body_mass_g"] = penguins_size["body_mass_g"].fillna(value = penguins_size["body_mass_g"].mean())

# %% value count of column sex
penguins_size.value_counts(["sex"])

# %% missing values in column Sex are to be imputed with “MALE”
penguins_size['sex'] = penguins_size['sex'].fillna('MALE')

# %% row with Sex as '.' is to be droped and check again with isna()
penguins_size.drop(penguins_size.index[penguins_size['sex'] == '.'], inplace=True)
penguins_size.value_counts(["sex"])

# or

# row 336 with Sex as '.' is to be droped and check again with isna()
#penguins_size.drop(axis = 0, inplace = True, index = 336)
#penguins_size.isna().sum()

# %% the presence of any duplicate rows is checked 
duplicated = penguins_size.duplicated()
print(duplicated.sum())

# %% Statistical insights
penguins_size.describe(include='all')

# %% unique values for the species
penguins_size['species'].value_counts()

# %% Find body mass mean for each species.
mean_bodymass = penguins_size.groupby('species')['body_mass_g'].mean()
mean_bodymass

# %% Relationship of the culmen length and sex of the penguins.
fig = plt.figure(figsize=(5,8))
ax= sns.boxplot(x=penguins_size['sex'], y=penguins_size['culmen_length_mm'])
plt.title('Culmen_length_mm')
plt.show()
# %% Shows us frequency distribution.
fig,axs = plt.subplots(1,4,figsize=(20,6))
axs[0].hist(penguins_size.culmen_depth_mm)
axs[0].set_title('culmen_depth_mm')
axs[0].set_ylabel('Frequency')
axs[1].hist(penguins_size.culmen_length_mm)
axs[1].set_title('culmen_length_mm')
axs[2].hist(penguins_size.flipper_length_mm)
axs[2].set_title('flipper_length_mm')
axs[3].hist(penguins_size.body_mass_g)
axs[3].set_title('body_mass_g')
plt.show()

# %% visualizing the probability density of flipper length
sns.kdeplot(penguins_size.flipper_length_mm,color='Cyan')
plt.show()

# %%  body masses of the penguins for each islands are seen
plt.figure(figsize=(8,5))
colors = ["cyan","lightblue", "darkblue"]
sns.barplot(x =penguins_size['island'],
y = penguins_size['body_mass_g'], palette = colors)
plt.title('Body Mass of Penguins for different Islands')
plt.show()

# %% Relationship between the number of penguins of specific species living in a particular island
pd.crosstab(penguins_size['island'], penguins_size['species']).plot.bar(color=('DarkBlue', 'LightBlue', 'Teal'))
plt.tight_layout()

# %% The number of observations of each species
sns.countplot('species',data=penguins_size, palette = "Oranges")
plt.show()

# %% Distribution of body mass in different islands
sns.violinplot(x='island',y='body_mass_g',data=penguins_size, palette="YlOrRd_r")
plt.title('Violin plot')

# %% Correlation matrix
corr = penguins_size.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, annot = True, cmap = "PuBu")
plt.title('Correlation Matrix')
plt.show()

# %%  visualizing the heatmap only with high values
sns.heatmap(corr[(corr > 0.8)],annot = True, cmap="PuBu")
# %%
