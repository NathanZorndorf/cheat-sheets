#------------------------------------------------
# Data Visualization with Pandas
#------------------------------------------------

#------------------------------------------------
# Tutorial:
# http://pandas.pydata.org/pandas-docs/stable/visualization.html
#------------------------------------------------


#--- general plot function
df.plot(x=None, y=None, kind='line', ax=None, subplots=False, use_index=True) # kind='plot_type'
df['Age'].plot(kind='hist')

'''
kind = 
‘line’ : line plot (default)
‘bar’ : vertical bar plot
‘barh’ : horizontal bar plot
‘hist’ : histogram
‘box’ : boxplot
‘kde’ : Kernel Density Estimation plot
‘density’ : same as ‘kde’
‘area’ : area plot
‘pie’ : pie plot
‘scatter’ : scatter plot
‘hexbin’ : hexbin plot

subplots : boolean
Make separate subplots for each column

use_index : Boolean 
Use index as ticks for x axis
'''

# Example: 
df.hist(layout=(n_cols, n_rows), color='r', alpha=0.8, figsize=(12,12), bins=40) # create a histogram of each column in the dataframe


# list of plot types to use with df.plot
df.plot.area()
df.plot.barh()
df.plot.density()
df.plot.hist()
df.plot.line()
df.plot.scatter()
df.plot.bar()
df.plot.box()
df.plot.hexbin()
df.plot.kde()
df.plot.pie()

#--- nicer styling
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#--- STYLING ---#
# http://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Show_colormaps
df.plot().bar(colormap='Greens')

plt.legend()