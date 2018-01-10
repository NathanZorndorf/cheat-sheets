#------------- SQLITE CHEAT SHEET --------------------#
import sqlite3

#-----------------------------------#
#------------- CONNECT -------------#
#-----------------------------------#

#---- Connecting to a database 
conn = sqlite3.connect('jobs.db')  # creates a Connection instance 

#-----------------------------------#
#------------- CURSOR --------------#
#-----------------------------------#

#---- Return a Cursor instance corresponding to the database we connected to
cursor = conn.cursor()

#---- Run a query 
query = "select * from recent_grads;" # SQL Query as a string
cursor.execute(query) # Execute the query, convert the results to tuples, and store as a local variable.
results = cursor.fetchall() # Fetch the full results set, as a list of tuples.
print(results[0:3]) # Display the first 3 results.

	#----- Alternatively, you don't need to make a cursor
	query = "select * from recent_grads;"
	conn.execute(query).fetchall()

#----- fetch one result
cursor.fetchone() # fetches result based on internal counter, which is incremeneted by 1 each call

#----- fetch many results
cursor.fetchmany(size=) # fetches result based on internal counter, which is incremeneted by size each call

#----- Save (commit) the changes
conn.commit()

#----- We can also close the connection if we are done with it.
# NOTE: Just be sure any changes have been committed or they will be lost.
conn.close()

#-----------------------------------#
#---- READ INTO PANDAS DATAFRAME ---#
#-----------------------------------#

#----- takes sql query string and connection object, and returns dataframe with rows and columns from query
sqlite3.read_sql_query(sql=, con=, index_col=) #sql=query, con=connection instance, index_col = column to set as index 
