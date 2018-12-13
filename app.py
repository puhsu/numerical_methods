"""
Main file of GUI application. All functions, related to
rendering are in this module. This file is basically
describes frontend of =adserver=
"""
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output

import util
import server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
)
app.config['suppress_callback_exceptions'] = True

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

    html.Hr(),
    html.H4('Plots'),
    html.Div(id='plots-container'),

    html.Div(id='pdf_intermediate', style={'display': 'none'})
])


@app.callback(
    Output('function-input-content', 'children'),
    [Input('function-input-tabs', 'value')]
)
def render_function_input(value):
    """Rendering input components for functions: ρ(x), S(t), z(t)
    """
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
            html.Div(id='file-validation', className='row mgn')
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
        ], className='mgn')


@app.callback(
    Output('plots-container', 'children'),
    [Input('function-input-tabs', 'value')]
)
def render_plots_frame(value):
    pdf_id = 'probability-density-' + value
    shows_id = 'shows-' + value
    threshold_id = 'threshold-' + value
    plan_id = 'diff-' + value

    return [
        html.Div([
            dcc.Graph(
                id=pdf_id,
                className='six columns',
            ),
            dcc.Graph(
                id=shows_id,
                className='six columns',
            ),
        ], className='row'),

        html.Div([
            dcc.Graph(
                id=threshold_id,
                className='six columns',
            ),
            dcc.Graph(
                id=plan_id,
                className='six columns',
            ),
        ], className='row'),
    ]


# PDF ploting
@app.callback(
    Output('probability-density-file', 'figure'),
    [Input('rho-file', 'contents')]
)
def plot_pdf_file(rho_file):
    return util.plot_lines(
        [util.parse_contents(rho_file)],
        ['PDF'],
        'Probability density',
    )


@app.callback(
    Output('probability-density-params', 'figure'),
    [Input('pdf-a', 'value'),
     Input('pdf-b', 'value')]
)
def plot_pdf_params(rho_a, rho_b):
    if rho_a and rho_b:
        return util.plot_lines(
            [util.tabulate_probability_density(rho_a, rho_b)],
            ['PDF'],
            'Probability density',
        )
    return {}


@app.callback(
    Output('shows-file', 'figure'),
    [Input('rho-file', 'contents'),
     Input('plan-file', 'contents'),
     Input('traffic-file', 'contents'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_shows_file(
        pdf_file, plan_file, traffic_file,
        x0, y0, beta, tau
):
    if pdf_file and plan_file and traffic_file:
        pdf = util.parse_contents(pdf_file)
        plan = util.parse_contents(plan_file)
        traffic = util.parse_contents(traffic_file)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        return util.plot_lines(
            [real_shows, plan, traffic],
            ['Real shows', 'Shows plan', 'Traffic'],
            title='Shows and traffic',
            xaxis='Time',
            yaxis='Shows count',
        )


@app.callback(
    Output('shows-params', 'figure'),
    [Input('pdf-a', 'value'),
     Input('pdf-b', 'value'),
     Input('plan-m', 'value'),
     Input('plan-n', 'value'),
     Input('plan-k', 'value'),
     Input('traffic-p', 'value'),
     Input('traffic-q', 'value'),
     Input('traffic-r', 'value'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_shows_params(
        pdf_a, pdf_b,
        plan_m, plan_n, plan_k,
        traffic_p, traffic_q, traffic_r,
        x0, y0, beta, tau
):
    if (pdf_a and pdf_b and plan_m and plan_n and plan_k and traffic_p and traffic_q and traffic_r
       and x0 is not None and y0 is not None and beta and tau):

        pdf = util.tabulate_probability_density(pdf_a, pdf_b)
        plan = util.tabulate_plan(plan_m, plan_n, plan_k, tau)
        traffic = util.tabulate_traffic(traffic_p, traffic_p, traffic_r, tau)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        return util.plot_lines(
            [real_shows, plan, traffic],
            ['Real shows', 'Shows plan', 'Traffic'],
            title='Shows and traffic',
            xaxis='Time',
            yaxis='Shows count',
        )
    return {}


@app.callback(
    Output('threshold-file', 'figure'),
    [Input('rho-file', 'contents'),
     Input('plan-file', 'contents'),
     Input('traffic-file', 'contents'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_threshold_file(
        pdf_file, plan_file, traffic_file,
        x0, y0, beta, tau
):
    if pdf_file and plan_file and traffic_file:
        pdf = util.parse_contents(pdf_file)
        plan = util.parse_contents(plan_file)
        traffic = util.parse_contents(traffic_file)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        # shows plot
        return util.plot_lines(
            [threshold],
            ['Threshold'],
            title='Threshold',
            xaxis='Time',
            yaxis='Threshold value',
        )


@app.callback(
    Output('threshold-params', 'figure'),
    [Input('pdf-a', 'value'),
     Input('pdf-b', 'value'),
     Input('plan-m', 'value'),
     Input('plan-n', 'value'),
     Input('plan-k', 'value'),
     Input('traffic-p', 'value'),
     Input('traffic-q', 'value'),
     Input('traffic-r', 'value'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_theshold_params(
        pdf_a, pdf_b,
        plan_m, plan_n, plan_k,
        traffic_p, traffic_q, traffic_r,
        x0, y0, beta, tau
):
    if (pdf_a and pdf_b and plan_m and plan_n and plan_k and traffic_p and traffic_q and traffic_r
       and x0 is not None and y0 is not None and beta and tau):

        pdf = util.tabulate_probability_density(pdf_a, pdf_b)
        plan = util.tabulate_plan(plan_m, plan_n, plan_k, tau)
        traffic = util.tabulate_traffic(traffic_p, traffic_p, traffic_r, tau)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        # shows plot
        return util.plot_lines(
            [threshold],
            ['Threshold'],
            title='Threshold',
            xaxis='Time',
            yaxis='Threshold value',
        )
    return {}


@app.callback(
    Output('diff-file', 'figure'),
    [Input('rho-file', 'contents'),
     Input('plan-file', 'contents'),
     Input('traffic-file', 'contents'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_diff_file(
        pdf_file, plan_file, traffic_file,
        x0, y0, beta, tau
):
    if pdf_file and plan_file and traffic_file:
        pdf = util.parse_contents(pdf_file)
        plan = util.parse_contents(plan_file)
        traffic = util.parse_contents(traffic_file)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        diff = plan
        diff[:, 1] -= real_shows[:, 1]
        return util.plot_lines(
            [diff],
            ['Difference'],
            title='Value',
            xaxis='Time',
            yaxis='S(t) - x(t)',
        )


@app.callback(
    Output('diff-params', 'figure'),
    [Input('pdf-a', 'value'),
     Input('pdf-b', 'value'),
     Input('plan-m', 'value'),
     Input('plan-n', 'value'),
     Input('plan-k', 'value'),
     Input('traffic-p', 'value'),
     Input('traffic-q', 'value'),
     Input('traffic-r', 'value'),
     Input('cauchy-x0', 'value'),
     Input('cauchy-y0', 'value'),
     Input('cauchy-beta', 'value'),
     Input('cauchy-tau', 'value')]
)
def plot_diff_params(
        pdf_a, pdf_b,
        plan_m, plan_n, plan_k,
        traffic_p, traffic_q, traffic_r,
        x0, y0, beta, tau
):
    if (pdf_a and pdf_b and plan_m and plan_n and plan_k and traffic_p and traffic_q and traffic_r
       and x0 is not None and y0 is not None and beta and tau):

        pdf = util.tabulate_probability_density(pdf_a, pdf_b)
        plan = util.tabulate_plan(plan_m, plan_n, plan_k, tau)
        traffic = util.tabulate_traffic(traffic_p, traffic_p, traffic_r, tau)

        real_shows, threshold = server.solve(pdf, plan, traffic, x0, y0, beta, tau)

        diff = plan
        diff[:, 1] -= real_shows[:, 1]
        diff[:, 1] = np.abs(diff[:, 1])
        return util.plot_lines(
            [diff],
            ['Difference'],
            title='Value',
            xaxis='Time',
            yaxis='S(t) - x(t)',
        )
    return {
    }


if __name__ == '__main__':
    app.run_server(host='localhost', debug=True)
