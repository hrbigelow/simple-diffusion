import fire
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.palettes import TolRainbow, interp_palette
from bokeh.core.enums import Dimensions
from streamvis import Server 
import numpy as np

def line_plot(schema, key, xcol_name, yaxis, **line_kwargs):
    cols = { v: [] for v in schema[key] }
    fig = figure(title=key, width=750, height=500, x_axis_label=xcol_name,
            y_axis_label=key, resizable=Dimensions.both)

    cit = iter(interp_palette(TolRainbow[20], len(cols)))
    cds = ColumnDataSource(cols, name=key)
    for y in cols.keys():
        if y != xcol_name:
            fig.line(x=xcol_name, y=y, legend_label=y, source=cds,
                    line_color=next(cit),
                    **line_kwargs)
    fig.legend.location = 'top_right'
    return fig

def vbar(schema, key, xcol_name):
    fig = figure(title=key, height=500, resizable=Dimensions.both)
    counts = schema[key]
    cols = { v: [] for v in schema[key] } 
    ycol_name = next(k for k in schema[key] if k != xcol_name)
    cds = ColumnDataSource(cols, name=key)
    fig.vbar(x=xcol_name, top=ycol_name, width=0.1, source=cds)
    return fig

def init_page(schema):
    # print(f'in init_page with {schema}')
    xbymu = line_plot(schema, 'xbymu', 'x', 'mu')
    xbysigma = line_plot(schema, 'xbysigma', 'x', 'sigma')
    loss = line_plot(schema, 'loss', 'step', 'loss', line_width=3)
    psamples = vbar(schema, 'psamples', 'x') 

    row1 = row(xbymu, xbysigma)
    row2 = row(loss, psamples)
    return row1, row2

def update_data(doc, run_data):
    # print(f'in update_page with run_data.keys={run_data.keys()}')
    # print('starting update')
    for cds_name in ('xbymu', 'xbysigma', 'psamples'):
        cds = doc.get_model_by_name(cds_name)
        if cds is not None and cds_name in run_data and len(run_data[cds_name]) > 0:
            cds.data = {}

    for cds_name, ent in run_data.items():
        cds = doc.get_model_by_name(cds_name)
        if cds is None:
            continue
        for step in sorted(ent.keys(), key=int):
            if cds_name in ('xbymu', 'xbysigma', 'psamples'):
                cds.data.update(ent[step])
            else:
                cds.stream(ent[step])
    # print('finished update')

def main(rest_host, rest_port, run_name):
    server = Server(run_name, rest_host, rest_port, init_page, update_data) 
    server.start()

fire.Fire(main)

