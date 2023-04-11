import fire
import numpy as np
from streamvis import Server
from bokeh.plotting import figure, gridplot
from bokeh.transform import linear_cmap
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from copy import deepcopy

def make_figure(figtype, key, fig_opts={}, **kwargs):
    cmap_range = kwargs.pop('cmap_range', None)
    fig = figure(title=key, **kwargs)
    colnames = 'xy' if cmap_range is None else 'xyz'
    col_map = { col: [] for col in colnames }
    cds = ColumnDataSource(col_map, name=key)
    vis_kwargs = { 'source': cds }
    if cmap_range is not None:
        low, high = cmap_range
        cmap = linear_cmap('z', palette='Viridis256', low=low, high=high)
        if figtype == 'multi_line':
            vis_kwargs['line_color'] = cmap 
        else:
            vis_kwargs['color'] = cmap

    if figtype == 'scatter':
        fig.scatter(x='x', y='y', **fig_opts, **vis_kwargs)
    elif figtype == 'line':
        fig.line(x='x', y='y', **fig_opts, **vis_kwargs)
    elif figtype == 'multi_line':
        fig.multi_line(xs='x', ys='y', **fig_opts, **vis_kwargs)
    return fig

scatter_plots = ('mu', 'log_sigma', 'psamples', 'rbf_centers')
color_plots = ('sigma_alphas', 'mu_alphas')
# keys = ('q',)

def init_page(schema):
    loss = make_figure('multi_line', 'loss', {'line_color': 'z'})
    # loss = make_figure('multi_line', 'loss')
    mu = make_figure('multi_line', 'mu', min_width=800)
    log_sigma = make_figure('scatter', 'log_sigma', cmap_range=(-5, -3))
    rbf_centers = make_figure('scatter', 'rbf_centers')
    mu_alphas = make_figure('scatter', 'mu_alphas', cmap_range=(0, 40))
    sigma_alphas = make_figure('scatter', 'sigma_alphas', cmap_range=(0, 40))
    psamples = make_figure('scatter', 'psamples', { 'size': 0.5 }, cmap_range=(0,40))

    g = gridplot(
            [
                [mu, rbf_centers, mu_alphas], 
                [loss, sigma_alphas, psamples]
                ],
            height=500, merge_tools=True)
    # q = scatter('q', 1900, 1000)
    return g

def append_multi_line(cds, entry):
    """
    Append step data
    entry:  (step => (column => [data...]))
    """
    agg = {}
    if len(entry) == 0:
        return

    # specially handle the 'z' index.  don't do any appending
    # transform entry into (column => mat[i, step])
    for step in sorted(entry.keys(), key=int):
        for col, data in entry[step].items():
            mat = agg.setdefault(col, [])
            mat.append(data)
    npagg = { col: np.array(mat).transpose(1,0) for col, mat in agg.items() }

    cds_data = cds.data
    for col, new_data in npagg.items():
        col_data = cds_data[col]
        if len(col_data) == 0:
            col_data.extend(new_data.tolist())
            # print(f'newly created col_data: {col_data}')
        else:
            for i, row in enumerate(col_data):
                row.extend(new_data[i])

    cds_data['z'] = viridis(40)
    cds.data.update(cds_data)
    # print(f'cds data = {cds.data}')

        
def update_data(doc, run_data):
    for key in scatter_plots + color_plots:
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
    append_multi_line(cds, entry)

server = Server('localhost', 8081, 'swissroll', init_page, update_data)
server.start()


