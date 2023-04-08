import fire
from streamvis import Server
from bokeh.plotting import figure 
from bokeh.transform import linear_cmap
from bokeh.layouts import row
from bokeh.models import ColumnDataSource

def scatter(key, width=500, height=450):
    fig = figure(width=width, height=height, title=key)
    cds = ColumnDataSource({'x': [], 'y': []}, name=key)
    fig.scatter(x='x', y='y', source=cds)
    return fig

def color_scatter(key, width, color_value_lo, color_value_hi):
    fig = figure(width=width, height=500, title=key)
    cds = ColumnDataSource({'x': [], 'y': [], 'z': []}, name=key)
    cmap = linear_cmap('z', palette='Viridis256', low=color_value_lo,
            high=color_value_hi, low_color='red', high_color='blue')
    fig.scatter(x='x', y='y', color=cmap, source=cds)
    return fig

def line(key, width):
    fig = figure(width=width, height=500, title=key)
    cds = ColumnDataSource({'x': [], 'y': []}, name=key)
    fig.line(x='x', y='y', source=cds)
    return fig

def multi_line(key, width):
    fig = figure(width=width, height=500, title=key)
    cds = ColumnDataSource({'x': [], 'y': []}, name=key)
    fig.multi_line(xs='x', ys='y', source=cds)
    return fig

scatter_keys = ('centers', 'sigmas', 'mu_alphas10')
mid_row = ('grid_mu10', 'grid_sigma10', 'sigma_alphas')
# keys = ('q',)

def init_page(schema):
    # q = scatter('q', 1900, 1000)
    # return q
    
    top = row(*(scatter(k, 450) for k in scatter_keys))
    grid_mu = multi_line('grid_mu10', 450)
    grid_sigma = color_scatter('grid_sigma10', 450, 0.45, 0.55)
    sigma_alphas = color_scatter('sigma_alphas', 450, 0, 40)

    mid = row(sigma_alphas, grid_mu, grid_sigma)
    bot = line('loss', 700)
    return top, mid, bot 

def update_data(doc, run_data):
    for key in scatter_keys + mid_row:
        cds = doc.get_model_by_name(key)
        if cds is None:
            continue
        entry = run_data[key]
        if len(entry) == 0:
            continue
        max_step = max(entry.keys(), key=int)
        cds.data = entry[max_step]

    cds = doc.get_model_by_name('loss')
    if cds is None:
        return
    entry = run_data['loss']
    for step in sorted(entry.keys(), key=int):
        cds.stream(entry[step])

server = Server('localhost', 8080, 'swissroll', init_page, update_data)
server.start()


