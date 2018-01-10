#----------------------- NUMPY CHEAT SHEET -------------------------#
import numpy as np

#--------- 2D ----------#
# Create 2D array
a = np.array([[1,2,3],
              [4,5,6]])

# index to reference entire first row
a[0] # output => array([1, 2, 3])

a[1, -2:] # output => index to reference last two column values in second row
