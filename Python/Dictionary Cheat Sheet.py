#-------------------- DICTIONARIES ---------------------#
#---- Make a dict using keys from a list
d.fromkeys()
	# Example :
	liquor_dict = {}
	liquor_dict.fromkeys(liquor_categories, None)

#---- Get value from key if key in dict
d.get()

#---- Check if a dict has a key
d.has_key()

#---- Returns list of key,value pairs as tuples 
d.items()

#---- Return a list of values in dict
d.values()

#---- Return a list of keys in dict\
d.keys()

#---- Returns an interator over the key,value items in dict
d.iteritems()

#-------------- METHODS
#---- sort items in a dict by returning a sorted list
sorted(d.items(), key=operator.itemgetter(1), reverse=True)