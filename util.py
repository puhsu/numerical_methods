import base64
import io
import functools

import numpy as np
import dash_core_components as dcc
import dash_html_components as html


##############################
# Layout helpers
##############################
def get_numeric_input(input_id, placeholder):
    return dcc.Input(
        input_id,
        placeholder=placeholder,
        type='number',
        inputmode='numeric',
        step=0.1,
        debounce=True,
        pattern=r'^[0-9]*\.{0,1}[0-9]*$'
    )


def get_file_upload(input_id, text):
    return dcc.Upload(
        id=input_id,
        children=html.Div([
            text,
            html.A('choose file')
        ]),
        className='upload',
        multiple=False,
    )


#############################
# Utility functions
##############################
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # TODO add checks for file
    stream = io.StringIO(decoded.decode('utf-8'))
    data = np.loadtxt(
        stream,
        delimiter=',',
        skiprows=1,
    )
    return data


def tabulate_probability_density(a, b, points=100):
    x = np.linspace(0, 1, points)
    fx = a * x * (b - x)
    return np.dstack([x, fx]).reshape(points, -1)


def tabulate_plan(m, n, k, tau, points=100):
    x = np.linspace(0, tau, points)
    fx = m * x + n * np.sin(k * x)
    return np.dstack([x, fx]).reshape(points, -1)


def tabulate_traffic(p, q, r, tau, points=100):
    x = np.linspace(0, tau, points)
    fx = p * x + q * np.cos(r * x)
    return np.dstack([x, fx]).reshape(points, -1)


##############################
# Plotting with plot.ly
##############################
def plot_lines(lines, names, title, xaxis=None, yaxis=None):
    return {
        'data': [
            {'x': line[:, 0], 'y': line[:, 1], 'type': 'line', 'name': name}
            for line, name in zip(lines, names)
        ],
        'layout': {
            'title': title,
            'xaxis': {'title': xaxis or ''},
            'yaxis': {'title': yaxis or ''},
        }
    }

##############################
# Other utils
##############################


class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)
