import seaborn as sns
%matplotlib inline

# Pictures and EXplanations of different plots 
http://seaborn.pydata.org/tutorial/distributions.html


#---- Scatter plot 
plt.scatter()

	# Example : plot a single point 
	plt.scatter([sepal_length_centroid[0]],[sepal_length_centroid[1]], marker='x', color='red', s=400)

	# Plot a set of points in a dataframe  
	sns.regplot(x, y, data=None)

#------------------------------------------
#----------- REGRESSION PLOTS -----------#
#------------------------------------------
#----- LINE PLOT 
sns.lmplot()

#------------------------------------------
#----------- DISTRIBUTION PLOTS ----------#
#------------------------------------------
#----- histogram / distribution plot 
sns.distplot()

	#----- Create a range to crop distribution to
	ds = sns.distplot(a=, bins=, hist_kws={'range':(0,200)})
	ds.set_xlim(0,200)

	#----- Overlay histograms
	# http://stackoverflow.com/questions/32899463/how-can-i-overlay-two-graphs-in-seaborn
	fig, ax = plt.subplots()
	for a in [x, y]:
	    sns.distplot(a, bins=range(1, 110, 10), ax=ax, kde=False)
	ax.set_xlim([0, 100])
	
	#----- Side by Side histograms
	# http://stackoverflow.com/questions/36362624/how-to-plot-multiple-histograms-on-same-plot-with-seaborn
	plt.hist([x, y], color=['r','b'], alpha=0.5)

#----- jointplot = scatter + 2 plots 
# kind: {'scatter' | 'reg'|  “resid” | “kde” | “hex”}
# use sns.JointGrid if you want another plot not listed
sns.jointplot(x=,y=,data=,kind=) 

	# Make a plot with a regression line of a different color
	sns.jointplot(x=concurrent_similar_median_start_price_series, y=end_price, kind='reg', ci=5, color='b',joint_kws={'line_kws':{'color':'cyan'}})

#---- pair plot = pairwise relationships across an entire dataframe
sns.pairplot(data=,hue=)

	# Example : only plot lower half of pairplot triangle
	g = sns.pairplot(iris)
	for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    	g.axes[i, j].set_visible(False)

#---- rugplot 
sns.rugplot(data=)

#---- kde plot
sns.kdeplot(data=)  # univariate density 
sns.kdeplot(data=, data2=) # plots a bivariate density
sns.kdeplot(data=, data2=, shade=True) # filled contours, no lines

#------------------------------------------
#----------- CATEGORICAL PLOTS -----------#
#------------------------------------------
#--- barplot = barplot is a general plot that allows you to aggregate 
sns.barplot() # categorical data based off some function, by default the mean
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std) #You can change the estimator object to your own function, that converts a vector to a scalar:

#---- countplot = same as barplot, just counting the number of occurences 
sns.countplot()

	# Example : horizontal count plot
	fig = plt.figure(figsize=(10,6))
	sns.countplot(y='vehicle_type_code1', data=df)

	
#--- boxplot 
sns.boxplot() 

#--- violin plots 
sns.violinplot() 

#--- strip plot = scatter plot where one variable is categorical, good complement to violin/box
sns.stripplot()
sns.stripplot(x="day", y="total_bill", data=tips)
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True) # adds jitter to better visualize 

#--- swarm plot
sns.swarmplot() 
sns.swarmplot(x="day", y="total_bill", data=tips)

#------------------------------------------#
#----------- MATRIX PLOTS -----------------#
#------------------------------------------#
#---- HEATMAP
sns.heatmap()
sns.heatmap(tips.corr(), cmap='RdBu_r') 

#--- CLUSTER MAP 
sns.clustermap()



#------------------------------------------#
#----------- GENERAL PLOT -----------------#
#------------------------------------------#
sns.factorplot()
sns.factorplot(x='sex',y='total_bill',data=tips,kind='bar') # 'kind' parameter decides type of plot

#----------- GENERAL GRIDS ---------------#
#--- pairgrid
sns.PairGrid(iris) # Just the Grid
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

#--- pariplot
sns.pairplot(iris,hue='species',palette='rainbow')
	
	

	# Example: create a histogram of each column in the dataframe
	# This just uses pandas built in plotting 
	df.hist(layout=(n_cols, n_rows), color='r', alpha=0.8, figsize=(12,12), bins=40) 



#--- facate grid = make a grid based on a feature 
g = sns.FacetGrid(tips, col="time", row="smoker") # Just the Grid

g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
g = g.map(plt.scatter, "total_bill", "tip").add_legend()

#--- joint grid
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)



#------------------------------------------#
#----------- PLOT FORMATTING --------------#
#------------------------------------------#


#--------------- FIGURES ----------#

#--- Figure size / figsize
fig = plt.figure(figsize=(9,9))
sns.barplot(x=gn.index, y=gn['Suppression Personnel'])


#---- Multiple Plots 
plt.figure(1)
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)
plt.figure(2)
sns.countplot(x='Embarked', data=titanic_df)


#--------------- SUBPLOTS ---------#

#---- Multiple subplots on one figure
fig, (axis1,axis2,axis3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=titanic_df, ax=axis2)
sns.barplot(x='Embarked', y='Survived', data=titanic_df, ax=axis3)


	# example : 2 rows and 3 cols -> need to pack your axes like so... 

	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(16,6))

	sns.regplot(st.iloc[:, 0], sub.iloc[:, 0], ax=ax1)
	sns.regplot(st.iloc[:, 0], sub.iloc[:, 1], ax=ax2)
	sns.regplot(st.iloc[:, 0], sub.iloc[:, 2], ax=ax3)
	sns.regplot(st.iloc[:, 0], sub.iloc[:, 3], ax=ax4)
	sns.regplot(st.iloc[:, 0], sub.iloc[:, 4], ax=ax5)



#---------------- AXES ------------#

#----------------------------------------------
# Documentation for axes properties/functions:
# http://matplotlib.org/api/axes_api.html
#----------------------------------------------

#---- Set X limit, Y Limit
g = sns.distplo()
g.set_ylim(bottom=None, top=None)
g.set_xlim(bottom=None, top=None)

#---- Set order of x axis
order = sorted(dff['Month'].unique())
sns.barplot(x=dff['Month'], y=dff[dff['Liquor Category']=='VODKA']['Profit (Dollars)'], order=order)

#---------- SET AXES/TITLES ---------------#
g = sns.barplot(x='Embarked', y='Survived', data=train_df[train_df['Sex']==1])
g.axes.set_title('Embarked vs Mean Survival Rate (for Males)')
g.set(xlabel='common xlabel', ylabel='common ylabel')

#--- Set a legend


	# Example
	sns.distplot(a=df[df['chain']==4].psoda, color='Yellow', label='Wendys')






#------------------------------------------#
#----------- PLOT STYLE -------------------#
#------------------------------------------#

#--------------------------------------------------
#http://seaborn.pydata.org/tutorial/aesthetics.html
#--------------------------------------------------


#---- Reset style parameters
sns.reset_orig()	 # Restore all RC params to original settings (respects custom rc).

#---- Set axis style 
sns.set(style=)
# style => dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}

#---- See current axis style
sns.axes_style() # returns current settings 

#---- rotate axes 
plt.xticks(rotation=60)
plt.xticks(rotation=55, horizontalalignment='right') 

#--- reset default stylings
sns.set()

#--- setting plot theme
sns.set_style() # 5 themes: darkgrid, whitegrid, dark, white, ticks
sns.set_style('whitegrid')

#--- configuring spines
sns.despine() # removes spines 
sns.despine(left=True)

#--- setting axis styling
sns.axes_style() # no arguments returns the current settings
sns.set_style("darkgrid", {"axes.facecolor": ".9"}) # can then set specific settings
# temporarily set style using 'with'
with sns.axes_style('darkgrid'):
	plt.subplot(211)
	sinplot()

#--- scale plot elements via context / set size
sns.set_context() # paper, notebook, talk, poster

	# Example :
	sns.set_context("talk")
	plt.figure(figsize=(8, 6))
	sinplot()

#--- Change font size 
sns.set(font_scale=1.5)

#--- set the color cycle using a seaborn palette
sns.set_palette()
sns.set_palette(palette=)

#--- get a list of colors defining a color palette
sns.color_palette() # use with sns.palplot() or sns.set_palette()
# Example:
sns.set_palette(sns.color_palette(palette='bright'))
# Matplotlib paletes can be specified as reversed palettes by appending “_r”
# to the name or as dark palettes by appending “_d” to the name. 

#---- Opacity
sns.displot(hist_kws={'alpha':1}) # Make completely opaque




#--------- COLOR PALETTES --------#

#--------------------------------------------------
# http://matplotlib.org/examples/color/colormaps_reference.html
# http://chrisalbon.com/python/seaborn_color_palettes.html
# http://seaborn.pydata.org/tutorial/color_palettes.html
#--------------------------------------------------


# Color Brewer library has a great set of palettes
sns.choose_colorbrewer_palette() #
sns.choose_colorbrewer_palette(data_type=) #data_type = {'sequential', 'diverging', 'qualitative'}


#------ QUALITATIVE 

sns.color_palette(palette=) 
# for qualitative color palettes, involving 6 or fewer factors
palette = deep, muted, pastel, bright, dark, colorblind
# more than 6 factors
sns.palplot(sns.color_palette('hls', n_colors=)) # n_colors =  the number of colors
sns.palplot(sns.color_palette('huls', n_colors=)) # keeps brightness even


#------ SEQUENTIAL 

# This kind of color mapping is appropriate when data range from 
# relatively low or unintersting values to relatively high or interesting values.
# used with kdeplot(), or corrplot()

sns.palplot(sns.color_palette("Blues"))
sns.palplot(sns.color_palette("BuGn_r")) # reversed lightness ramp


#------ DIVERGING 

# These are used for data where both large low and high values are interesting. 
sns.palplot(sns.color_palette("BrBG", 7))
sns.palplot(sns.color_palette("coolwarm", 7))


#------------------------------------------#
#----------- EXPORTING A FILE -------------#
#------------------------------------------#
g = sns.barplot()
g.savefig("barplot_output.png")



#------------------------------------------#
#---------------- EXAMPLES  ---------------#
#------------------------------------------#

#---- Make a heatmap to discover correlation

mean_corr = affair[[x for x in affair.columns]].corr()
# Set the default matplotlib figure size:
plt.rcParams['figure.figsize']=(9,7)
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(mean_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(mean_corr, mask=mask)
# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)
# If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
plt.show()




