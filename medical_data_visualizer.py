# we will visualize and make calculations from medical examination data using matplotlib, seaborn, and pandas
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')
df.head()
df.shape
df.info()


#Create the overweight column in the df variable
df['overweight'] = df['weight']/((df['height']/100)*(df['height']/100))
df.shape

#Normalize data by making 0 always good and 1 always bad. 
df['overweight']=(df['overweight']>25).astype(int)
df['cholesterol'] = (df['cholesterol']>1).astype(int)
df['gluc'] = (df['gluc']>1).astype(int)

#draw a cat plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.(
    df_cat = sorted(['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.melt(df, id_vars='cardio', value_vars=df_cat)

    # Get the figure for the output
    fig = sns.catplot(x='variable', col='cardio', hue='value', kind='count', data=df_cat).set_axis_labels('variable', 'total')

    fig.savefig('catplot.png')
    return fig
    plt.show()
draw_cat_plot()

# draw the heat map
def draw_heat_map():
  sns.heatmap(df)
draw_heat_map()

# Clean the data
df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) &
(df['height'] >= df['height'].quantile(0.025)) &
(df['height'] <= df['height'].quantile(0.975)) &
(df['height'] >= df['height'].quantile(0.025)) &
(df['height'] <= df['height'].quantile(0.975))]

# Calculate the correlation matrix
corr = df_heat.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr) #return a array of zeros with the same shape as corr 
mask[np.triu_indices_from(mask)] = True # indices for the upper-triangle of arr.

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize = (8, 8))

#plot the correlation matrix using the method provided by the seaborn 
ax = sns.heatmap(corr,vmin=0.0, vmax=0.15, center=0, annot=True, fmt='.2f', annot_kws=None, linewidths=0, square=True, mask=mask)
plt.title('Correlation Matrix')
plt.show()

