"""Main file of GUI application. All functions, related to
rendering are in this module. This file is basically
describes frontend of adserver
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State

import util
import server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
)
app.config['suppress_callback_exceptions'] = True

# store grid functions in this dictionary
# this solution is limited to one user apps
# and cannot be used for multiple users
grid_store = {}


app.layout = html.Div([
    html.H1('Ad server'),

    html.Div('''Implementation of a system that simulates the behavior of an
    advertising server, described as a system of differential
    equations.'''),

    html.Hr(),
    html.H4('Functions input'),

    dcc.Tabs(id="function-input-tabs", value='params', children=[
        dcc.Tab(label='Load as file', value='file'),
        dcc.Tab(label='Input parameters', value='params'),
    ]),

    html.Div(id='function-input-content'),

    html.Hr(),
    html.H4('Parameters'),

    util.get_numeric_input('cauchy-x0', 'x₀'),
    util.get_numeric_input('cauchy-y0', 'y₀'),
    util.get_numeric_input('cauchy-beta', 'β'),
    util.get_numeric_input('cauchy-tau', 'τ'),
    util.get_numeric_input('auto-beta0', 'auto β₀'),
    util.get_numeric_input('auto-beta1', 'auto β₁'),

    html.Br(),
    html.Button(id='build-plot', n_clicks_timestamp='0', children='plot', className='mgn-l mgn-t'),
    html.Button(id='auto', n_clicks_timestamp='0', children='auto', className='mgn-l'),
    html.Br(),
    html.P(
        '''When using auto mode, you don\'t need to specify x₀, y₀ and β. You only
        need to specify the region for β
        ''',
        className='mgn',
    ),

    html.Hr(),
    html.H4('Plots'),
    html.Div(id='plots-container'),
])


@app.callback(
    Output('function-input-content', 'children'),
    [Input('function-input-tabs', 'value')]
)
def render_function_input(value):
    """Rendering input components for functions.

    Based on chosen input type construct the requested input components
    for ρ(x), S(t), z(t) (probability density, shows plan and real traffic)
    """
    global grid_store
    grid_store = {}
    if value == 'file':
        return html.Div([
            html.Div([
                html.P('Load csv file in following format: x,f(x)'),
                html.Div([
                    html.Div(
                        util.get_file_upload('rho-file', 'Probability density ρ(x) '),
                        className='four columns',
                    ),
                    html.Div(
                        util.get_file_upload('plan-file', 'Shows plan S(t) '),
                        className='four columns',
                    ),
                    html.Div(
                        util.get_file_upload('traffic-file', 'Traffic z(t) '),
                        className='four columns'
                    ),
                ], className='row'),
            ], className='row mgn'),
            html.Div(id='file-validation', className='row mgn'),
            html.Div(id='placeholder_file', style={'display': 'none'}),
        ])
    if value == 'params':
        return html.Div([
            html.P('Parameters for ρ(x) = ax(b - x)'),
            util.get_numeric_input('pdf-a', 'a'),
            util.get_numeric_input('pdf-b', 'b'),

            html.P('Parameters for S(t) = mt + n·sin(kt)'),
            util.get_numeric_input('plan-m', 'm'),
            util.get_numeric_input('plan-n', 'n'),
            util.get_numeric_input('plan-k', 'k'),

            html.P('Parameters for z(t) = pt + q·cos(rt)'),
            util.get_numeric_input('traffic-p', 'p'),
            util.get_numeric_input('traffic-q', 'q'),
            util.get_numeric_input('traffic-r', 'r'),

            html.Br(),
            html.Div(id='placeholder_param', style={'display': 'none'}),
        ], className='mgn')


@app.callback(
    Output('placeholder_file', 'children'),
    [Input('rho-file', 'contents'),
     Input('plan-file', 'contents'),
     Input('traffic-file', 'contents')]
)
def save_functions_from_files(pdf_file, plan_file, traffic_file):
    if pdf_file and plan_file and traffic_file:
        grid_store['pdf'] = util.parse_contents(pdf_file)
        grid_store['plan'] = util.parse_contents(plan_file)
        grid_store['traffic'] = util.parse_contents(traffic_file)
    return ''


@app.callback(
    Output('placeholder_param', 'children'),
    [Input('pdf-a', 'value'),
     Input('pdf-b', 'value'),
     Input('plan-m', 'value'),
     Input('plan-n', 'value'),
     Input('plan-k', 'value'),
     Input('traffic-p', 'value'),
     Input('traffic-q', 'value'),
     Input('traffic-r', 'value'),
     Input('cauchy-tau', 'value')]
)
def save_functions_from_params(
        pdf_a, pdf_b,
        plan_m, plan_n, plan_k,
        traffic_p, traffic_q, traffic_r,
        tau,
):
    params = [pdf_a, pdf_b, plan_m, plan_n, plan_k, traffic_p, traffic_q, traffic_r, tau]
    if all(p is not None for p in params):
        grid_store['pdf'] = util.tabulate_probability_density(pdf_a, pdf_b)
        grid_store['plan'] = util.tabulate_plan(plan_m, plan_n, plan_k, tau)
        grid_store['traffic'] = util.tabulate_traffic(traffic_p, traffic_q, traffic_r, tau)
    return ''


@app.callback(
    Output('plots-container', 'children'),
    [Input('build-plot', 'n_clicks_timestamp'),
     Input('auto', 'n_clicks_timestamp')],
    [State('cauchy-x0', 'value'),
     State('cauchy-y0', 'value'),
     State('cauchy-beta', 'value'),
     State('cauchy-tau', 'value'),
     State('auto-beta0', 'value'),
     State('auto-beta1', 'value')]
)
def render_plots(simple, auto, x0, y0, beta, tau, beta0, beta1):
    if int(simple) > int(auto):
        return render_plots_simple(x0, y0, beta, tau)
    else:
        return render_plots_auto(beta0, beta1, tau)


def render_plots_simple(x0, y0, beta, tau):
    if (
            all(k in grid_store for k in ['pdf', 'plan', 'traffic']) and
            all(p is not None for p in [x0, y0, beta, tau])
    ):
        # x(t), y(t)
        real_shows, threshold = server.solve(
            grid_store['pdf'],
            grid_store['plan'],
            grid_store['traffic'],
            x0, y0, beta, tau,
        )

        # C_1(beta), C_2(beta)
        crit1 = server.crit1(real_shows, threshold, grid_store['pdf'], x0, tau)
        crit2 = server.crit2(real_shows, grid_store['plan'])

        # |S(t) - x(t)|
        diff = np.zeros_like(real_shows)
        diff[:, 0] = real_shows[:, 0]
        diff[:, 1] = np.abs(grid_store['plan'][:, 1] - real_shows[:, 1])

        return [
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [real_shows, grid_store['plan'], grid_store['traffic']],
                        ['Shows', 'Plan', 'Traffic'],
                        'Shows',
                        yaxis='value',
                        xaxis='time',
                    )
                ),
            ], className='row'),
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [threshold],
                        ['threshold y(t)'],
                        'Threshold',
                        yaxis='value',
                        xaxis='time',
                    ),
                    className='six columns',
                ),
                html.Div(
                    children=f'C₁ = {crit1:.2f}  C₂ = {crit2:.2f}',
                    className='six columns criterions',
                )
            ], className='row'),
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [grid_store['pdf']],
                        ['pdf'],
                        'Probability density',
                        yaxis='value',
                        xaxis='x',
                    ),
                    className='six columns',
                ),
                dcc.Graph(
                    figure=util.plot_lines(
                        [diff],
                        ['diff'],
                        '|S(t) - x(t)|',
                        xaxis='time',
                        yaxis='value',
                    ),
                    className='six columns',
                )
            ], className='row')
        ]
    return 'Input all parameters and hit plot button to see relevant plots'


def render_plots_auto(beta0, beta1, tau):
    if all(p is not None for p in [beta0, beta1, tau]):
        # initializations
        beta = np.linspace(beta0, beta1, 20)
        x0 = 0
        y0 = 0.5
        loss = []

        # find optimal beta
        for b in beta:
            real_shows, threshold = server.solve(
                grid_store['pdf'],
                grid_store['plan'],
                grid_store['traffic'],
                x0, y0, b, tau,
            )

            # C_1(beta), C_2(beta)
            crit1 = server.crit1(real_shows, threshold, grid_store['pdf'], x0, tau)
            crit2 = server.crit2(real_shows, grid_store['plan'])
            loss.append(crit1 + 10 * crit2)

        loss = np.array(loss)
        best = beta[np.argmin(loss)]

        print('Best beta=', best)

        # now find some optimal points
        nx, ny = (5, 5)
        criterions = np.zeros((nx, ny))

        xs = np.linspace(0, grid_store['traffic'][0, 1], nx)
        ys = np.linspace(0, 1, ny)

        for i in range(nx):
            for j in range(ny):
                x = xs[i]
                y = ys[j]
                real_shows, threshold = server.solve(
                    grid_store['pdf'],
                    grid_store['plan'],
                    grid_store['traffic'],
                    x, y, best, tau,
                )

                # C_1(beta), C_2(beta)
                crit1 = server.crit1(real_shows, threshold, grid_store['pdf'], x, tau)
                crit2 = server.crit2(real_shows, grid_store['plan'])
                criterions[i, j] = crit1 + 10 * crit2

        i, j = np.unravel_index(np.argmin(criterions, axis=None), criterions.shape)
        best_x = xs[i]
        best_y = ys[j]

        # get 3 best x, y and plot solutions for them
        idxs = np.dstack(
            np.unravel_index(np.argsort(criterions.ravel())[:3], criterions.shape)
        )[0]

        solutions = []
        for x, y in idxs:
            real_shows, threshold = server.solve(
                grid_store['pdf'],
                grid_store['plan'],
                grid_store['traffic'],
                xs[x], ys[y], best, tau,
            )
            solutions.append(real_shows)

        # compute with best params
        real_shows, threshold = server.solve(
            grid_store['pdf'],
            grid_store['plan'],
            grid_store['traffic'],
            best_x, best_y, best, tau,
        )

        diff = np.zeros_like(real_shows)
        diff[:, 0] = real_shows[:, 0]
        diff[:, 1] = np.abs(grid_store['plan'][:, 1] - real_shows[:, 1])

        return [
            # Best shows function
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [real_shows, grid_store['plan'], grid_store['traffic']],
                        ['Shows', 'Plan', 'Traffic'],
                        f'Shows for best x₀={best_x:.2f} y₀={best_y:.2f} β={best:.2f}',
                        yaxis='value',
                        xaxis='time',
                    )
                ),
            ], className='row'),
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [threshold],
                        ['threshold y(t)'],
                        f'Threshold for best x₀={best_x:.2f} y₀={best_y:.2f} β={best:.2f}',
                        yaxis='value',
                        xaxis='time',
                    ),
                    className='six columns',
                ),
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': beta, 'y': loss, 'type': 'line', 'name': 'loss'}
                        ],
                        'layout': {
                            'title': 'Loss',
                            'xaxis': {'title': 'beta'},
                            'yaxis': {'title': 'C₁ + 10C₂'},
                        }
                    },
                    className='six columns',
                )
            ], className='row'),
            html.Div([
                dcc.Graph(
                    figure=util.plot_lines(
                        [grid_store['pdf']],
                        ['pdf'],
                        'Probability density',
                        yaxis='value',
                        xaxis='x',
                    ),
                    className='six columns',
                ),
                dcc.Graph(
                    figure=util.plot_lines(
                        [diff],
                        ['diff'],
                        '|S(t) - x(t)|',
                        xaxis='time',
                        yaxis='value',
                    ),
                    className='six columns',
                )
            ], className='row'),

            html.Div([
                dcc.Graph(
                    figure={
                        'data': [
                            {
                                'x': np.repeat(xs, nx),
                                'y': np.tile(ys, ny),
                                'z': criterions.flatten(),
                                'mode': 'markers',
                                'type': 'scatter3d',
                                'marker': {
                                    'size': 12,
                                    'opacity': 0.8,
                                },
                            }
                        ],
                        'layout': {
                            'title': 'Loss',
                            'xaxis': {'title': 'x'},
                            'yaxis': {'title': 'y'},
                            'zaxis': {'title': 'loss'}
                        }
                    },
                    className='six columns'
                ),
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': grid_store['plan'][:, 1], 'y': s[:, 1], 'type': 'line', 'name': f'x₀={x}, y₀={y}'}
                            for s, (x, y) in zip(solutions, idxs)
                        ],
                        'layout': {
                            'title': '(S, x) plots for 3 best points',
                            'xaxis': {'title': 'S(t)'},
                            'yaxis': {'title': 'x(t)'},
                        }
                    },
                    className='six columns'
                )
            ])
        ]


if __name__ == '__main__':
    app.run_server(host='localhost', debug=True)
