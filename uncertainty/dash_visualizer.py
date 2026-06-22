from typing import Dict, List
from itertools import chain
import json
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import numpy as np
from itertools import chain

from typing import Dict, List
from uncertainty.utils import load_json


color_matching_dict = {"CE_std": "#D94350",
                       "TV_gt_exps_p(1-p)_sum_CE_var": "#8B278C",
                       "TV_CE_p(1-p)_sum_CE_var": "#117DBF", 
                       "mc_dropout": "#25AAE2",
                       "one_net": "#14A697",
                       "ddu": "#F2A341",
                       "huq": "#652D90", 

                       "pred_experts": "#A492D1",
                       "oracle": "#CA6702",
                       "TV_pred_exps_p(1-p)_sum_CE_var": "#469609", 
                       "TV_mixed_loss_p(1-p)_sum_mixed_loss_var": "#94D2BD", 
                       "mcmc":"#600996",
                       "gt_experts": "#EE9B00", 
                       "test_augmentations": "#BB3F03",  
                       "conform_set_pca45": "#D19F92",
                       "TV_oracle_p(1-p)_sum_CE_var": "#64B2CA",
                       "CE_p(1-p)": "#EBBB7F",
                       "CE_entropy": "#21D0DA", 
                       "gt_experts_std": "#4FA82C",
                       "gt_experts_all_p(1-p)": "#7FEBAA",
                       "gt_experts_entropy": "#0C44BC", 
                       "entropy_CE_sum_CE": "#650996",
                       "entropy_gt_exps_sum_CE": "#D14F27",
                       }


def calculate_aucs(accuracies_dict: dict, throwaway_rates_dict: dict,
                   min_accuracy: float | None = None):
    '''
    Calculates areas between the accuracy-rejection curve and the bottom horizontal line
    given by min_accuracy.

    @param accuracies_dict: {unc estimation method name: accuracies_list}
    @param throwaway_rates_dict: {unc estimation method name: throwaway_rates_list (x axis values)}
    '''
    def normalize_accuracies(accuracies: list[float], min_acc: float):
        '''
        Fits accuracies to [0, 1] range (for convinient AUC calculation).
        '''
        return [(accuracy - min_acc) / (1.0 - min_acc) for accuracy in accuracies]

    aucs = {}
    if min_accuracy is None:
        min_accuracy = min(list(chain(accuracies_dict.values())))

    for method_name, accuracies in accuracies_dict.items():
        accuracies = normalize_accuracies(accuracies, min_accuracy)
        auc = np.trapezoid(y=accuracies[: int(len(accuracies) * 0.8) + 1], x=throwaway_rates_dict[method_name][: int(len(throwaway_rates_dict[method_name]) * 0.8) + 1])
        aucs[method_name] = round(auc, 6)
        print(f"Area under {method_name} curve: {aucs[method_name]}")

    return aucs


def plot_rejection_curves(data_dict: Dict, init_accuracy, display_stds: bool = False):
    '''
    Plots accuracy-rejection curves (dependence of accuracies on the left test set part
    after the most uncertain samples are thrown away).

    TODO: display std button
    '''
    def choose_color(
            used_colors_ids: List,
            colorpalette: tuple[int, ...] = ("#809609", "#026374", "#099396", "#94D2BD", "#965709",
                                             "#EE9B00", "#CA6702", "#96096C", "#600996")
    ):
        if len(used_colors_ids) == len(colorpalette):
            used_colors_ids = []
        color_id = np.random.choice(list(set(range(len(colorpalette))) - set(used_colors_ids)))
        used_colors_ids.append(color_id)

        color_hex = colorpalette[color_id]
        color_rgb_values = [int(color_hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]
        return color_rgb_values, used_colors_ids
    
    traces = [go.Scatter(x=[0, 1], 
                         y=[init_accuracy, init_accuracy], 
                         name='random', mode='lines',
                         line = dict(color='#646464', width=1.7, dash='dot')), 
              go.Scatter(x=[0, 1. - init_accuracy, 1], 
                         y=[init_accuracy, 1, 1], 
                         name='ideal', mode='lines',
                         line = dict(color='#099669', width=1.7, dash='dot'))
    ]

    accuracies_dict, throwaway_rates_dict = {}, {}
    used_colors_ids = []

    layout = go.Layout(template=None, title=dict(text="Rejection curves"),
        xaxis=dict(title='Throwaway rate', range=[0., 0.5]), yaxis=dict(title='Accuracy', range=[init_accuracy - 0.02, 1.02]))

    for label, data in data_dict.items():
        # color_rgb, used_colors_ids = choose_color(used_colors_ids)
        color_rgb = [int(color_matching_dict[label].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]

        throwaway_rates = sorted(list(map(float, list(data.keys()))))
        means, stds = [], []
        for throwaway_rate in throwaway_rates:
            means.append(data[str(throwaway_rate)][0])
            stds.append(data[str(throwaway_rate)][1])

        if display_stds:
            # Maybe use zip(means, stds) instead?
            std_upper = [means[i] + std for i, std in enumerate(stds)]
            std_lower = [means[i] - std for i, std in enumerate(stds)]

            traces.append(go.Scatter(
                x=throwaway_rates + throwaway_rates[::-1],  # x, then x reversed
                y=std_upper + std_lower[::-1],  # upper, then lower reversed,
                fill='tozerox', line={"width": 0}, showlegend=False,
                fillcolor=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.4)",))

        traces.append(go.Scatter(
            x=throwaway_rates,
            y=means,
            name=label,
            line={
                "color": f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)",
                "dash": "solid",
                "width": 1.7
            }
        ))

        accuracies_dict[label] = means
        throwaway_rates_dict[label] = throwaway_rates

    aucs = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0)
    return traces, layout, aucs


def beautify_fig4illustrations(fig):
    fig.update_layout(plot_bgcolor='white', 
        xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 0.1),
        yaxis = dict(tickmode = 'linear', tick0 = 0.72, dtick = 0.04),
        width=1000, height=500
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        range = [0.72, 1.02],
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    path2save_plot = "uncertainty_estimation/results/rej_curves.pdf"
    fig.write_image(path2save_plot)

    return fig

def run_dash(config):
    rej_curves_data_dir = config["path2save_rc_data"]

    paths = {}
    for method in config["UE_methods"].keys():
        path2rej_curves_data = f"{rej_curves_data_dir}/{method}.json"
        paths[method] = path2rej_curves_data

    init_accuracy = load_json(path2rej_curves_data)["0.0"][0]

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        html.H1("Welcome to the accuracy-rejection curves analyzer!",
                style={'font-size': 20, 'width': '100%', 'margin': '10pt',
                       'textAlign': 'center', 'font': "Courier New"}),

        ########
        html.Div(children=[
            html.Label('Rejection curves types:'),
            dcc.Checklist(
                id='rejection_curve_mode',
                options=list(paths.keys()),
                labelStyle={'font-size': 13},
                ),
            ],
            style={'width': '15%', 'display': 'inline-block'}),
        html.Div(dcc.Graph(id='rejection_curve', figure=go.Figure()),
                 style={'width': '74%', 'display': 'inline-block'}),
        html.Div(id='auc'),
        html.Br(),
    ])

    @app.callback(
            Output('rejection_curve', 'figure'),
            Output('auc', 'children'),
            [Input('rejection_curve_mode', 'value')]
    )
    def update_graph(rejection_curve_mode):
        uncertainty_types = {}
        if rejection_curve_mode is not None:
            for mode in rejection_curve_mode:
                uncertainty_types[mode] = load_json(paths[mode])
        
        traces, layout, aucs = plot_rejection_curves(uncertainty_types, init_accuracy, display_stds=False)

        # fig = go.Figure({'data': traces, 'layout': layout})
        # beautify_fig4illustrations(fig)
        
        message_about_aucs = "The areas above the rejection curves (x10^3): \n"
        left_acc = 1. - init_accuracy
        for label, auc in aucs.items():
            message_about_aucs += f"{label}: {round(0.8 - auc - (left_acc * left_acc / 2), 6) * 1000}; "
        return {'data': traces, 'layout': layout}, message_about_aucs

    # update_graph(list(paths.keys()))
    # exit()

    app.run(debug=True, port=8012)
