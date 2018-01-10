Jupyter Notebooks Cheat Sheet


https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

#--------- HOTKEYS --------# 
Ctrl + Shift + -  =>  will split the current cell into two from where your cursor is.

Shift + J/K   => to select multiple cells 
Shift + M  	  => merge all the selected cells.

# Jupyter will display that variable without the need for a print statement.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#-------- Settings --------#

# Change the Maxmimum allowable columns you can view
pd.set_option('display.max_columns', None)
# OR
with pd.option_context('display.max_columns', None): # temporarily 

# this line makes plots prettier on mac retina screens. If you don't have one it shouldn't do anything.
%config InlineBackend.figure_format = 'retina'

# See all currently declared variables, modules, etcl
%whos
