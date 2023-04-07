import fire
from streamvis import Server as SVServer 
from bokeh.plotting import figure 
from bokeh.models import ColumnDataSource

def init_page(schema):
    fig = figure(width=750, height=500, title='Swiss Roll')
    sw = schema['swiss_roll']
    cds = ColumnDataSource({'x': [], 'y': []}, name='swiss_roll')
    fig.scatter(x='x', y='y', source=cds)
    return fig

def update_data(doc, run_data):
    cds = doc.get_model_by_name('swiss_roll')
    entry = run_data['swiss_roll']
    for val in entry.values():
        cds.data.update(val)

sv = SVServer('swissroll', 'localhost', 8080, init_page, update_data)
sv.start()


