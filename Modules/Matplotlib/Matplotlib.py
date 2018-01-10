import matplotlib.pyplot as plt
%matplotlib inline # for use in jupyter notebooks


#---- Make a figure
fig, ax = plt.subplots(1, 1)

ax.plot(x, norm_distr.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.xlim(upper, lower)
plt.title('Normal Distribution PDF')


#--------------------------------------------------#
#------------- FIGURE, SUBPLOTS, AXES -------------#
#--------------------------------------------------#
#--------------------- FIGURE ---------------------#
plt.figure()

	# num	1	number of figure
	# figsize	figure.figsize	figure size in in inches (width, height)
	# dpi	figure.dpi	resolution in dots per inch
	# facecolor	figure.facecolor	color of the drawing background
	# edgecolor	figure.edgecolor	color of edge around the drawing background
	# frameon	True	draw figure frame or not


#--- Get the current figure class instance 
fig = plt.gcf() 

#--------------------- SUBPLOTS -------------------#
# With subplot you can arrange plots in a regular grid. You need to specify the number of rows and columns and the number of the plot. Note that the gridspec command is a more powerful alternative.
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,4));
	ax1.hist(x=df['sellingStatus.bidCount'], bins=50)
	ax1.set_xlim((0,80))

#--------------------- AXES -----------------------#
# Axes are very similar to subplots but allow placement of plots at any location in the figure. So if we want to put a smaller plot inside a bigger one we do so with axes. 

#--- Get the current `~matplotlib.axes.Axes` class instance on the current figure matching the given keyword args, or create one.
ax = plt.gca()


#--------------------------------------------------#
#--------------------- SETTINGS -------------------#
#--------------------------------------------------#

#---- set ticks
plt.xticks()

	# Example :
	plt.xticks(np.linspace(-4,4,9,endpoint=True))

	# Example : set numerical values for ticks, and set values to appear as labels
	plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])


#--------------------------------------------------#
#--------------------- SUBPLOTS -------------------#
#--------------------------------------------------#
plt.subplot(nrows, ncols, plot_number, figsize=)


#-------------------- LABELS ----------------------#

# Add title to subplots
ax.set_title('Standard Normal Distribution')

	# Example : 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Standard Normal Distribution')


# X labels / Y labels
ax.set_xlabel('label') # with subplot
# OR 
plt.xlabel('Value') # without subplot



#--- Legend 
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
plt.legend(loc=, frameon=False) # you can just call it after you've made some plots, make sure the 'legend' kwarg is set to your label

	# Example : legend for two graphs on one plot 
	fix, ax = plt.subplots(1, 1)
	ax.plot(x, t_5.pdf(x), 'r-', lw=2, alpha=0.6, label='n=5')
	ax.plot(x, t_100.pdf(x), 'b-', lw=2, alpha=0.6, label='n=100')
	plt.legend()


#-------------------- LAYOUT ----------------------#

# Arrange subplots nicely
fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

# Adjust spacing of subplots 
subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots

#---- Plot two graphs on same figure 

#---- Rotate labels
plt.xticks(rotation=80)

	# Example
	sns.barplot(x=grouped.index.tolist(), y=grouped['Profit (Dollars)'])
	plt.xticks(rotation=80)


#--------------------------------------------------#
#------------ TYPES OF PLOTS / GRAPHS -------------#
#--------------------------------------------------#

#--------- LINE ---------# 
plt.plot(list_1, list_2)

#---- Vertical Line : adds vertical line across the axes 
plt.axvline(constant_to_plot, color='r')

#--------- SCATTER ------# 
plt.scatter(x,y)

#-------- HISTOGRAM -----#
plt.hist(data)

	# Create a range to crop distribution to
	plt.hist(x=, bins=, range=(min_range, min_range))

#-------- BOX PLOT ------#
plt.boxplot(data,vert=True,patch_artist=True);   


#-------- BAR PLOT -----#

#--------------------------------------------------#
#------------------------ STYLE -------------------#
#--------------------------------------------------#

#---- Set color of spines 
ax.spines['top'].set_color('none')

#---- Set position of spines 
ax.spines['location'].set_position(('position_type', amount))

# Position Type:
#	‘outward’ : place the spine out from the data area by the specified number of points. (Negative values specify placing the spine inward.)
#	‘axes’ : place the spine at the specified Axes coordinate (from 0.0-1.0).
#	‘data’ : place the spine at the specified data coordinate.
# Amount : 
#	‘center’ -> (‘axes’,0.5)
# 	‘zero’ -> (‘data’, 0.0)

	 # Example
	ax.spines['bottom'].set_position(('data',0))

#------------------- COLOR ------------------------#
# Color Maps : https://www.labri.fr/perso/nrougier/teaching/matplotlib/

# Set each plot to a different color
color = iter(plt.cm.gist_rainbow(np.linspace(0,1,len(df.columns))))
for i in range(len(grouped_summed.columns)):
    c = next(color)
    plt.plot(a, color=c)

#--------------------------------------------------#
#--------------- COMMON WORKFLOWS -----------------#
#--------------------------------------------------#

# Create a new figure of size 8x6 points, using 100 dots per inch
plt.figure(figsize=(8,6), dpi=80)

# Create a new subplot from a grid of 1x1
plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# Plot cosine using blue color with a continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# Plot sine using green color with a continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

# Set x limits
plt.xlim(-4.0,4.0)

# Set x ticks
plt.xticks(np.linspace(-4,4,9,endpoint=True))

# Set y limits
plt.ylim(-1.0,1.0)

# Set y ticks
plt.yticks(np.linspace(-1,1,5,endpoint=True))

# Legend
	# Example: Create legend that is outside if figure
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # Shrink current axis by 20%
	legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis

	# Set Font size
	for label in legend.get_texts():
	    label.set_fontsize('large')

	# Set legend line widths
	for legobj in legend.legendHandles:
	    legobj.set_linewidth(3.0)

# Save figure using 72 dots per inch
savefig("../figures/exercice_2.png",dpi=72)
	# Save Figire with legend included 
	plt.savefig('./images/want_to_feel_relative_frequency', bbox_extra_artists=(legend,), bbox_inches='tight')


# Tight layout 
fig.tight_layout()

# Show result on screen
plt.show()



#--------------------------------------------------#
#------------------- EXAMPLES ---------------------#
#--------------------------------------------------#









