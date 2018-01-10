# Plotly & Cufflinks 
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # requires version >= 1.9.0

# For Notebooks
init_notebook_mode(connected=True)

import cufflinks as cf

df.iplot()
df.iplot(kind=, x=, y=, mode=, size=)

'''
iplot kind = 

scatter
bar
box
spread
ratio
heatmap
surface
histogram
bubble
candle
'''