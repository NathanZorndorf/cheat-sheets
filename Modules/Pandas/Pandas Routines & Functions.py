import pandas as pd
import numpy as np

#------------------------------------- DOCUMENTATION ---------------------------------------#

#--------- API Reference Documentation -------------#

http://pandas.pydata.org/pandas-docs/stable/api.html

#--------------------------------------- SETTINGS ------------------------------------------#
pd.set_option("max_columns", None)


#---------------------------------------- SERIES -------------------------------------------#

s = pd.Series(data=None, index=None, dtype=None, name=None)

#--------------------- SELECTING --------------------#
s.iloc[] # indexing with integer label
s.loc[] # indexing with string label
s[0:5] # select first 5 rows. This works even if you have a custom index 

#------------------ ORGANIZE / SORT -----------------#
Series.reindex(index=None) 
# index = New labels / index to conform to. Preferably an Index object to avoid duplicating data

Series.sort_index() 


pd.concat() # for combining a series to a dataframe, 
			# or concatinating multiple dataframes on to each other
bank_stocks = pd.concat(objs=[bac, c, gs], axis=1, keys=['BAC', 'C', 'GS'])

#---- minimum  / maximum
pd.argmin() # index of the minimum value
pd.argmax() # index of the maximum value 

#---- count the unique values
pd.value_counts(normalize=False,sort=True,ascending=False,bins=None,dropna=True) # Returns object containing counts of unique values.

Series.nunique() # counts number of unique values in a series 

#---- Unique values 
Series.unique() # Return np.ndarray of unique values in the object. 

#---- Rename a series
s.rename("different", inplace=True) # scalar, changes Series.name
s.rename(lambda x: x ** 2)  # function, changes labels (indeces)
s.rename({1: 3, 2: 5})  # mapping, changes labels (indeces)


#------------ ATTRIBUTES
#---- Name 
s.name

#---- Index values
s.index 
s.index.tolist() 








#------------------------------------- DATAFRAME ---------------------------------------#

#----------------------------------------------------#
#---------------- CREATE A DATAFRAME ----------------#
#----------------------------------------------------#

#---- From a dictionary
df1 = pd.DataFrame(d.items(), columns=['key_col','val_col'])


#----------------------------------------------------#
#--------------------- BASIC INFO -------------------#
#----------------------------------------------------#
#---- Get info / summary of dataframe
df.info() # Concise summary of a DataFrame.


#----------------------------------------------------#
#--------------------- SELECTING --------------------#
#----------------------------------------------------#
# Operation			Syntax			Result
# ---------			------			-------
df[col]				# Select column, returns Series
df.loc[label] 		# Select row by label, returns Series
df.iloc[loc] 		# Select row by integer location, returns Series
df[5:10] 			# Select a set of rows with all columns, returns	DataFrame
df[bool_vec] 		# Select rows by boolean vector, returns DataFrame

df.loc[25] # selects the row with the index LABEL of 25

df.iloc[25] # selects the 25th row 

df.iloc[0:10] # selects first 10 rows

df.loc[1100,'age'] # selects the values at the row with the index label of 1100, and the column 'age'

df.iloc[0:5, 0:3] # selects first 5 rows and 3 columns 

df.ix[:,['int_col', 'float_col']] # selects all rows, and a subset of the columns
# OR
df[['int_col', 'float_col']]

#--- Filtering with .isin()
dfnr = df[df['value']==-1] # -1 means no rating 
no_value_df = dfnr['subject_id'].value_counts() # get the UNIQUE subject_id's for the users who had -1 values into a list
df_filtered = df[~df['subject_id'].isin(no_value_df)] # keep only the rows of df_long where the subject_id was NOT(~) in the no_value_df list 

#--- conditional indexing / masking 

df[df['float_col'] > 0.15]
# OR
df[(df['float_col'] > 0.15) & (df['int_col'] > 2)]
# OR
df[~(df['float_col'] > 0.1)]

#---- Masking based on several conditions
Crit1 = df.AAA <= 5.5
Crit2 = df.BBB == 10.0
Crit3 = df.CCC > -40.0
crit = Crit3 & Crit2 & Crit1
df[crit]


#---- getting a subset of the dataframe 
df.where()
df['Name'].where(df['Cost'] > 3).dropna()

df = df[df['Bronze'] != df['Total']]
# OR 
df = df[df.Bronze != df.Total]


#---- dropping rows in a dataframe based on a mask
df.drop()

	# Example 
	missing_nuclei_rows = df[df['Bare_Nuclei'] == '?'].index
	df = df.drop(missing_nuclei_rows)
	# OR
	df = df[df['Bare_Nuclei'] == '?']


#---- getting a cross-section of the dataframe
pd.xs(key=, axis=, level=)

#---- Getting a random sample of the data
df.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)


#----------------------------------------------------#
#----------- MANIPULATING COLUMNS -------------------#
#----------------------------------------------------#
#---- Rename a column
df.rename(columns={'int_col' : 'some_other_name'}, inplace=True)
# OR
df.columns = ['col_1_name','col_2_name']  # another way to rename
# OR
df.rename(columns={list(df.columns)[0]:'new_col_0_name', list(df.columns)[1]:'new_col_0_name'}, inplace=True)

#---- Rename a column level name
df.columns.names = 'col_level_name'
#---- rename the index level name
df.index.names = 'index_level_name'

#---- Delete a column
df.drop(labels=,axis=0,level=None,inplace=False) 

	# EXample: drop columns using integer label indexing 
	df.drop(df.columns[i], axis=1) # where i is a number 
	# OR 
	df = df.iloc[:, i:] # where i is a number


# OR
del df['col_name']
# OR
popped_col = df.pop('col_name')


#---- Insert a column 
df['new_col'] = df['old_col'] * 2 # gets inserted at end of datagrame
# OR
df.insert(loc=2,column='new_col',value=df['old_col']) # insert column at specified location 


#---- Rearange columns 
df = df[['mean', '0', '1', '2', '3']]



#------------------ ORGANIZE / SORT -----------------#

#---- sort values : sort by values
df.sort_values(by=,axis=0,ascending=True,inplace=False)
	#----- Example:
	df.sort_values(by='Norm_Nutr_Index', axis=0, ascending=False, inplace=True)

#---- sort index : sort by index 
df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last')


#---- find the top n values in a series
df['col_name'].nlargest()

	# example : 
	df.groupby(by='Store Number')['Profit (Dollars)'].sum().nlargest(10)

#----------------- CHANGE INDEXING ------------------#

#---- reset index 
df.reset_index(level=None, drop=False, inplace=False)
# drop => False : Creates a new column with the current index values
#	   => True : drops old index values 

#---- set index
df.set_index(keys, drop=True, append=False, inplace=False)
# keys : column label or list of column labels / arrays
# drop => False : Keep the column(s) in the dataframe, and also use it/them as the index
#      => True : Delete column(s) to be used as the new index


#----------- New Columns = f(Existing Columns) -------#

#---- Calculate a new column of data 
df.assign(**kwargs) # assign always returns a copy of the data
  #---- Example:
  iris = iris.assign(sepal_ratio = lambda x: (x['SepalWidth']/x['SepalLength'])) # creates new column called 'sepal_ratio'

# Multiple columns as a function of a single column
df4['twice'],df4['thrice'] = zip(*df4['int_col'].map(two_three_strings))

# SIngle column as a function of multiple columns
df5['sum_col'] = df5.apply(sum_two_cols, axis=1)



#----------------------------------------------------#
#------------- MISSING VALUES -----------------------#
#----------------------------------------------------#
#---- Check for NaN / Null values
df.isnull() # returns the dataframe or series with True for NaN values, False otherwise
	
	# Example : Count the number of missing values
	df.isnull().values.sum() 	# number of total null values in dataframe
	# OR
	df.isnull().sum() 	# number of null values per column

	# Example : drop the rows that contain nan in certain columns
	df.dropna(axis=0, subset=labels['col_1'], inplace=True)

	# Example : find the row that contains a missing value in 'income' column
	df.loc[df['income'].isnull(), :]

	# Example : returns only the values in the Series that are NaN
	age = titanic_survival["age"]
	age_null_true = age[pd.isnull(age)]
	age_null_count = len(age_null_true) # number of missing items 

	# Example : returns only the values in a Series that are NOT NaN
	good_ages = titanic_survival[pd.isnull(titanic_survival["age"]) == False]['age']

#---- Creatre a heatmap of missing values 
sns.heatmap(data=train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

	
#---- Drop NaN / Null values
df.dropna(axis='index',how='any',inplace=False, subset=None)
# axis => 'index' or 'columns' or tuple or list
# how = {'any','all'} => any: if any NA values are present, drop that label. all: if all values are NA, drop that label
# subset: array-like: labels along other axis to consider e.g. if you are dropping rows, these would be columns to include
	# Example: drop all columns with NaN values
	drop_na_columns = titanic_survival.dropna(axis='columns')
	# Example : drop all ROWS where columns 'sex' or 'age' have NaN
	new_titanic_survival = titanic_survival.dropna(axis='index', subset=['age','sex'])

#---- Replace or Fill NaN / Null values 
df.fillna()

	# Example : 
	df.groupby(by='chain')['psoda'].apply(lambda x: x.fillna(x.mean())) # fill missing values in 'psoda' column with mean of 'psoda' column FOR EACH CHAIN 

	# Example:
	df3 = df.copy()
	df3['float_col'].fillna(df3['float_col'].mean()) # fill na values with mean 

	# Example:
	train_df['Age'] = train_df[['Age', 'Pclass']].apply(fill_age, axis=1) 
	
	def fill_age(x):
	    Age = x[0]
	    Pclass = x[1]
	    
	    if(pd.isnull(Age)):
	        if(Pclass == 1):
	            Age = mean_age_class_1
	        elif(Pclass == 2):
	            Age = mean_age_class_2
	        else:
	            Age = mean_age_class_3
	            
	    return Age
    

#----------------------------------------------------#
#------------- TRANSFORM DATA -----------------------#
#----------------------------------------------------#

#---- astype : transform / change type of data in dataframe
df1['DateIndex'].astype(dtype=str)
	
	# Example 
	df['Market Capitalization'] = df['Market Capitalization'].astype(dtype=float)

	# Example
	df[['col2','col3']] = df[['col2','col3']].apply(pd.to_numeric)



#---- get dummies : useful for converting categorical data to numerical data
pd.get_dummies(data=, drop_first=True) 



#---- map values to something else
map()
	
	# Example : replace all 'False.' values with 0 and everything else with 1 in the 'Churn?' column of the dataframe
	df['Churn?'] = df['Churn?'].map(lambda x: 0 if x=='False.' else 1) 



#---- df.replace()
df.replace(o_replace=None, value=None, inplace=False, axis=None) # replace to_replace with value for entire dataframe

	# Exapmle: replace any positve categorical with 1 
	true = ["yes",
	        "True",
	        "True."]
	df.replace(to_replace=true, value=1, inplace=True)



#--------------- STRING MANIPULATION ----------------#

#---- String replace 
df['Description'].str.replace('macbook air 11.*' , 'macbook air 11')
# OR
df.replace({'\n': '<br>'}, regex=True) # replace '\n' with <br> in entire dataframe
# OR 
df.replace(to_replace='Unknown', value=np.nan)




#----------------------------------------------------#
#------------- FINDING VALUES -----------------------#
#----------------------------------------------------#
#---- Find minimum value in a dataframe / Series
df.min(axis=None,skipna=True,numeric_only=None)
#---- Find index of first occurence of minimum over requested axis 
df.idxmin(axis=0,skipna=True)
	# axis  => 0 : get the index value for the max value in each column  
	#		=> 1 : get the column value for the max value in each row

#---- Find maximum
df.max()
#---- Find index of max
df.idxmax() 
	# Example : get the index value for the max value in the 'Gold' column
	df['Gold'].idxmax(axis=0)

# Filter data in a dataframe, splitting it into two (or more) dataframes
s_largest = set(largest_stores.index.values)
s_smallest = set(smallest_stores.index.values)
s_smallest = s_smallest.difference(s_largest)
smallest_stores.loc[list(s_smallest), :]

# Filter data to include only values in a list of values 
df[df.apply(lambda x: x['gender'] in [1,2], axis=1)] 	# Only keep data in dataframe if values in 'gender' column are 1 or 2



#----------------------------------------------------#
#------------------- JOINING  -----------------------#
#----------------------------------------------------#
#---- concatinating 
pd.concat() # for combining a series to a dataframe, 
			# or concatinating multiple dataframes on to each other
bank_stocks = pd.concat(objs=[bac, c, gs], axis=1, keys=['BAC', 'C', 'GS'])

#---- join 
pd.merge(df, other, on='str_col', how='inner') # how = inner, outer, left, right

#----------------------------------------------------#
#--------------- SETTING A VALUE --------------------#
#----------------------------------------------------#
d.sales[d.sales==24] = 100 # replace a value
# OR
d.sales = d.sales.replace(23, 24) # replace a value

#---- Set a value at a certain location
d.ix[row_index, 'col_name'] = 99

	# Example : 
	d.loc[d.sales == 12, 'sales'] = 99 # set a value at a location

	# Example : Replace  missing 'psoda' observations with the average price of soda per chain
	mask = df.psoda.isnull() & df.chain==1
	missing_indeces = df[mask].index.values
	df.loc[missing_indeces, 'psoda'] = BK_psoda_mean
	# OR
	df.groupby(by='chain')['psoda'].apply(lambda x: x.fillna(x.mean()))


#---- apply a function to every cell in a dataframe
df.applymap()

	# Example 
	df.applymap(np.sqrt)


#---- where : use of if-else idiom in pandas 
df.where()

	# Example : 
	df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True,False] * 2})
	df.where(df_mask,-1000)
	# OR 
	df['logic'] = np.where(df['AAA'] > 5,'high','low')


#----------------------------------------------------#
#----------------- MAP, APPLY -----------------------#
#----------------------------------------------------#
#---- APPLY 
df.apply(func,axis=0)	# The apply is a pretty flexible function which, as the name suggests, 
			# applies a function along any axis of the DataFrame.
	# axis => 'index' : apply function to each column
	#      => 'columns' : apply function to each row 

	# Example : use apply on a column 
	df['State Bottle Cost'] = df.apply(lambda x: float(x['State Bottle Cost'].replace('$','')), axis=1)

	# Example
	df.ix[:,['int_col','float_col']].apply(np.sqrt)

	# Example :  apply function using several columns
	df['newcolumn'] = df.apply(lambda x: fxy(x['A'], x['B']), axis=1)
	# OR
	df['avg'] = df[['RottenTomatoes','RottenTomatoes_User']].apply(func=np.mean,axis=1)


#---- MAP 
df.map() # The map operation operates over each element of a Series.

	# Example 
	df['str_col'].dropna().map(lambda x : 'map_' + x)


#---- APPLY MAP 
df.applymap() # can be used to apply the function to each element of the dataframe

# vectorized string operations
df['E'] = np.sqrt(df['A'])


#----------------------------------------------------#
#---------------- PIVOT TABLES ----------------------#
#----------------------------------------------------#
df.pivot_table(data=,values=None,index=None,columns=None,aggfunc='mean',fill_value=None,dropna=True)
# index : Keys to group by on the pivot table index.
# values : column to aggregate, optional (it's the values that the aggfunc is performed on)
# columns : Keys to group by on the pivot table column.  
# aggfunc specifies the calculation we want to perform 
	# Example : calculate the mean age for each passenger class
	passenger_age = titanic_survival.pivot_table(index='pclass', values='age')
	# Example : calculate sum of several columns 
	port_stats = titanic_survival.pivot_table(index='embarked', values=['fare','survived'], aggfunc=np.sum)


#----------------------------------------------------#
#-------------------- GROUPBY -----------------------#
#----------------------------------------------------#
grouped = df['float_col'].groupby(df['str_col'])
grouped.mean()

	# EXAMPLE: 
	# group by embarked, and get the mean for survived passengers for each value in Embarked
	titanic_df.groupby(by='Embarked').agg({'Survived': np.average}) 

#---- Groupby a dataframe and keep other ungrouped columns
df.groupby(['col1','col2','col3'])['col to aggregate'].sum() \
														.reset_index() \
														.sort_values('Profit (Dollars)', ascending=False) 

#--- Use a groupby as a mask 
df.loc[df.groupby("item")["diff"].idxmin()]

#----------------------------------------------------#
#-------------------- DATETIME ----------------------#
#----------------------------------------------------#
# Tutorial:
# http://nbviewer.jupyter.org/github/fonnesbeck/Bios8366/blob/master/notebooks/Section2_2-Data-Wrangling-with-Pandas.ipynb

# Datetime string format:
# http://strftime.org/


# Convert a column from string format to datetime 
# http://stackoverflow.com/questions/26763344/convert-pandas-column-to-datetime
df['DateTime'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

#--- Sort by DateTime


# Get hour from datetime object column
# http://stackoverflow.com/questions/25129144/pandas-return-hour-from-datetime-column-directly
df['Month'] = df['DateTime'].apply(lambda x: x.month)


# Get year from datetime column (series)
df['DateTime'].dt.hour 		# must add .dt in front of series


# get Hour from timestamp 
df['hour'] = df['timestamp'].map(lambda x: x.to_pydatetime().hour)

# Convert datetime from one format to another
# This should be done after you've donverte a column from string to datetime  
df['DateTime'] = df['DateTime'].dt.strftime('%-m/%-d/%Y %-H:%M')

# Delete trailing zeros from dates like '01/01/2017'
df['DateTime'] = df['DateTime'].dt.strftime('%-m/%-d/%Y') # Mac
df['DateTime'] = df['DateTime'].dt.strftime('%#m/%#d/%Y') # Windows

#----------------------------------------------------#
#--------------- MULTIINDEXING ----------------------#
#----------------------------------------------------#
#---- getting a cross-section of the dataframe / return data for level of a multiindex 
df.xs(key=, axis=, level=) # level can be a number or a string for the index label 
# EXAMPLE
df = pd.DataFrame({'row' : [0,1,2],
                   'One_X' : [1.1,1.1,1.1],
                   'One_Y' : [1.2,1.2,1.2],
                   'Two_X' : [1.11,1.11,1.11],
                   'Two_Y' : [1.22,1.22,1.22]}); 
df.set_index('row')
df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns])

     One        Two      
       X    Y     X     Y
row                      
0    1.1  1.2  1.11  1.22
1    1.1  1.2  1.11  1.22
2    1.1  1.2  1.11  1.22

# Select only X columns
in[6]: df.xs(key='X', axis=1, level=1) 

out[6]:
	One	Two
row		
0	1.1	1.11
1	1.1	1.11
2	1.1	1.11


#---- breaks/pivots a multindexed table into a single indexed table 
df.unstack(level=)


#------------- MATH on a DATAFRAME --------#
#---- creates a rolling window to apply functions onto 
df.rolling(window=) 

'''
window = size of moving window
min_periods = Minimum number of observations in window required to have a value (otherwise result is NA).
center = 
on = For a dataframe,c olumns on which to calculate the rolling window, rather than the index 
axis = 

By default, the result is set to the right edge of the window. 
This can be changed to the center of the window by setting center=True.
'''


#------------- ATTRIBUTES ---------------#
#---- Shape : gives axis dimensions 
df.shape

#---- columns
df.columns
	#----- Example : get names of columns as a list 
	df.columns.tolist()

#---- index
df.index

#---- size : number of elements in the dataframe 
df.size 

#---- dtypes : return the data types in this object 
df.dtypes
	# Example
	types = fandango_films.dtypes # returns the data types as a Series	
	float_columns = types[types.values == 'float64'].index # filter data types to just floats, index attributes returns just column names	
	float_df = fandango_films[float_columns] # use bracket notation to filter columns to just float columns

#---- axes : Return a list with the row axis labels and column axis labels 
df.axes


# -----------------------------------------#
# -------- STATISTICS ---------------------#
# -----------------------------------------#
df.describe() # provides quick stats on all columns 
df.corr() 

# -----------------------------------------#
# --------- EXPORTING ---------------------#
# -----------------------------------------#

# write to csv | pd.to_csv
output.to_csv("name.csv", index=, quoting=)





