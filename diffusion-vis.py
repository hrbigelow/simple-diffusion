from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider
from streamvis import Receiver
import numpy as np

def update(all_data, step, source):
    new_data = all_data[step]
    if 'mu' in new_data:
        # source.data.update(new_data['mu'])
        source.stream(new_data['mu'])

port=1234
xdata = np.linspace(0, 1, 100)
ydata = np.random.randn(100)

doc = curdoc()
source = ColumnDataSource(data=dict(x=[], y=[]))
recv = Receiver(port, doc, update, source) 


plot = figure(height=300)
plot.multi_line(xs='x', ys='y', source=source)

# slider = Slider(start=0, end=100, value=0, step=1)
doc.add_root(plot)

recv.start()
