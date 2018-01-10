

#--- endswith() 
string.endswith('.csv') # returns True if the string ends with the string passed in as the argument

#--- iterating through a dictionary
d = {(1,2): 3, (4,5): 6}

for k, v in d:
	print(k)
	print(v)

# A new built-in function, enumerate(), will make certain loops a bit clearer. 
# enumerate(thing), where thing is either an iterator or a sequence, 
# returns a iterator that will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.
L = [10,11,12,13,14]

for i, item in enumerate(L):
    # ... compute some result based on item ...
    print(i)
    print(item)

#------ replace characters in a string ------ #
# http://stackoverflow.com/questions/3939361/remove-specific-characters-from-a-string-in-python
import re
line = re.sub('[!@#$]', '', line)
# OR
# Python 3
translation_table = dict.fromkeys(map(ord, '!@#$'), None)
unicode_line = unicode_line.translate(translation_table)
# OR
# Python 3
unicode_line = unicode_line.translate({ord(c): None for c in '!@#$'})
#OR
new_headlines_words = []
for word in headlines_words:
    for char in word:
        if char in bad_chars:
            word = word.replace(char,'')   
    new_headlines_words.append(word)


#---- map : apply a function to each item in a list
map(function, iter)
	# Example : make a list containing the length of each strings in a list of strings 
	map(len, strings)

#---- filter : filter / only keep elements that satisfy a a predicate. 
filter(pred, iter)
	# Example : 
	filter(is_even, range(100))

#--------------------- ERRORS --------------------#
 # List of errors: https://www.tutorialspoint.com/python/python_exceptions.htm

#---- isinstance
isinstance(object, type)


#---- Assert : assert tests an expression, and if the result comes up false, an exception is raised.
# assert is like a raise-AssertionError-if-not statement. 
# An expression is tested, and if the result comes up false, an AssertionError exception is raised.
assert statement


#---- Exception handling 
#---- finally 
try:

except:

finally:

#---- else 
try:

except:

else: 

try:
    fh = open(file_path, "r+")
except Exception as e:
    print "type of exception is: ", e.__class__.__name__
# OR
try:
    fh = open(file_path, "r+")
except IOError, argument:
    print argument





