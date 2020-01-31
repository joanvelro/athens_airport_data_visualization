import pandas as pd
import matplotlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

""" Read passengers data and obtain dataframe
"""
if 1 == 1:
    data_path = "D:\\INDRA\\ATM-TT-data\\Athens_data\\"

    df1 = pd.read_csv(data_path + 'passengers_16_17.csv', sep=' ', index_col=0)
    df2 = pd.read_csv(data_path + 'passengers_17_18.csv', sep=' ', index_col=0)
    df3 = pd.read_csv(data_path + 'passengers_18_19.csv', sep=' ', index_col=0)

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df3.reset_index(inplace=True)

    df = pd.merge(df3, df2, how='inner', on='month', left_index=True, suffixes=('_3', '_2'))

    df.drop(['var_18_19', 'var_18_19.1', 'var_18_19.2', 'var_17_18', 'var_17_18.1', 'var_17_18.2',
             'domestic_18_2', 'inter_18_2', 'total_18_2'], axis=1, inplace=True)

    df = pd.merge(df, df1, how='inner', on='month', left_index=True, suffixes=('__', '_1'))

    df.drop(['var_16_17', 'var_16_17.1', 'var_16_17.2',
             'domestic_17_1', 'total_17_1', 'inter_17_1'], axis=1, inplace=True)

    df.rename(columns={'domestic_18_3': 'domestic_18', 'inter_18_3': 'inter_18',
                       'total_18_3': 'total_18', 'total_17__': 'total_17',
                       'inter_17__': 'inter_17', 'domestic_17__': 'domestic_17'}, inplace=True)

    time_index = pd.date_range('2016-01-01', '2020-01-01', freq='M')

    index_2016 = np.array(['2016-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2017 = np.array(['2017-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2018 = np.array(['2018-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2019 = np.array(['2019-{}'.format(i + 1) for i in np.arange(0, 12)])

    index_ = np.concatenate([index_2016, index_2017, index_2018, index_2019])

    df_passengers = pd.DataFrame(data=[],
                                 index=index_,
                                 columns=['domestic', 'international', 'total'])
    df_passengers_var = pd.DataFrame(data=[],
                                     index=np.arange(0, 12),
                                     columns=['var_domestic_18_19', 'var_international_18_19', 'var_total_18_19'])

    for i in np.arange(0, 12):
        df_passengers.loc['2016-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_16']].values[0]
        df_passengers.loc['2017-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_17']].values[0]
        df_passengers.loc['2018-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_18']].values[0]
        df_passengers.loc['2019-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_19']].values[0]
        df_passengers.loc['2016-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_16']].values[0]
        df_passengers.loc['2017-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_17']].values[0]
        df_passengers.loc['2018-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_18']].values[0]
        df_passengers.loc['2019-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_19']].values[0]
        df_passengers.loc['2016-{}'.format(i + 1), ['total']] = df.loc[i, ['total_16']].values[0]
        df_passengers.loc['2017-{}'.format(i + 1), ['total']] = df.loc[i, ['total_17']].values[0]
        df_passengers.loc['2018-{}'.format(i + 1), ['total']] = df.loc[i, ['total_18']].values[0]
        df_passengers.loc['2019-{}'.format(i + 1), ['total']] = df.loc[i, ['total_19']].values[0]

    df_passengers.set_index(time_index, inplace=True)
    df_passengers.to_csv(data_path + 'passengers_agg_16_19.csv', index=True)
    df_passengers_total = df_passengers.resample('Y').sum()

    for i in np.arange(0, 12):
        df_passengers_var.loc[i, ['var_domestic_18_19']] = (df.loc[i, ['domestic_19']].values[0] -
                                                            df.loc[i, ['domestic_18']].values[0]) / \
                                                           df.loc[i, ['domestic_18']].values[0]
        df_passengers_var.loc[i, ['var_international_18_19']] = (df.loc[i, ['inter_19']].values[0] -
                                                                 df.loc[i, ['inter_18']].values[0]) / \
                                                                df.loc[i, ['inter_18']].values[0]
        df_passengers_var.loc[i, ['var_total_18_19']] = (df.loc[i, ['total_19']].values[0] -
                                                         df.loc[i, ['total_18']].values[0]) / \
                                                        df.loc[i, ['total_18']].values[0]
""" Read flights data and obtain dataframe
"""
if 1 == 1:
    data_path = "D:\\INDRA\\ATM-TT-data\\Athens_data\\"

    df1 = pd.read_csv(data_path + 'flights_16_17.csv', sep=' ', index_col=0)
    df2 = pd.read_csv(data_path + 'flights_17_18.csv', sep=' ', index_col=0)
    df3 = pd.read_csv(data_path + 'flights_18_19.csv', sep=' ', index_col=0)

    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df3.reset_index(inplace=True)

    df = pd.merge(df3, df2, how='inner', on='month', left_index=True, suffixes=('_3', '_2'))

    df.drop(['var_18_19', 'var_18_19.1', 'var_18_19.2', 'var_17_18', 'var_17_18.1', 'var_17_18.2',
             'domestic_18_2', 'inter_18_2', 'total_18_2'], axis=1, inplace=True)

    df = pd.merge(df, df1, how='inner', on='month', left_index=True, suffixes=('__', '_1'))

    df.drop(['var_16_17', 'var_16_17.1', 'var_16_17.2',
             'domestic_17_1', 'total_17_1', 'inter_17_1'], axis=1, inplace=True)

    df.rename(columns={'domestic_18_3': 'domestic_18', 'inter_18_3': 'inter_18',
                       'total_18_3': 'total_18', 'total_17__': 'total_17',
                       'inter_17__': 'inter_17', 'domestic_17__': 'domestic_17'}, inplace=True)

    df_flights_var = pd.DataFrame(data=[],
                                  index=np.arange(0, 12),
                                  columns=['var_domestic_18_19', 'var_international_18_19', 'var_total_18_19'])

    time_index = pd.date_range('2016-01-01', '2020-01-01', freq='M')

    index_2016 = np.array(['2016-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2017 = np.array(['2017-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2018 = np.array(['2018-{}'.format(i + 1) for i in np.arange(0, 12)])
    index_2019 = np.array(['2019-{}'.format(i + 1) for i in np.arange(0, 12)])

    index_ = np.concatenate([index_2016, index_2017, index_2018, index_2019])

    df_flights = pd.DataFrame(data=[],
                              index=index_,
                              columns=['domestic', 'international', 'total'])

    for i in np.arange(0, 12):
        df_flights.loc['2016-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_16']].values[0]
        df_flights.loc['2017-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_17']].values[0]
        df_flights.loc['2018-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_18']].values[0]
        df_flights.loc['2019-{}'.format(i + 1), ['domestic']] = df.loc[i, ['domestic_19']].values[0]
        df_flights.loc['2016-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_16']].values[0]
        df_flights.loc['2017-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_17']].values[0]
        df_flights.loc['2018-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_18']].values[0]
        df_flights.loc['2019-{}'.format(i + 1), ['international']] = df.loc[i, ['inter_19']].values[0]
        df_flights.loc['2016-{}'.format(i + 1), ['total']] = df.loc[i, ['total_16']].values[0]
        df_flights.loc['2017-{}'.format(i + 1), ['total']] = df.loc[i, ['total_17']].values[0]
        df_flights.loc['2018-{}'.format(i + 1), ['total']] = df.loc[i, ['total_18']].values[0]
        df_flights.loc['2019-{}'.format(i + 1), ['total']] = df.loc[i, ['total_19']].values[0]

    index_months = df_flights.index
    df_flights.set_index(time_index, inplace=True)
    df_flights.to_csv(data_path + 'flights_agg_16_19.csv', index=True)
    df_flights_total = df_flights.resample('Y').sum()

    for i in np.arange(0, 12):
        df_flights_var.loc[i, ['var_domestic_18_19']] = (df.loc[i, ['domestic_19']].values[0] -
                                                            df.loc[i, ['domestic_18']].values[0]) / \
                                                           df.loc[i, ['domestic_18']].values[0]
        df_flights_var.loc[i, ['var_international_18_19']] = (df.loc[i, ['inter_19']].values[0] -
                                                                 df.loc[i, ['inter_18']].values[0]) / \
                                                                df.loc[i, ['inter_18']].values[0]
        df_flights_var.loc[i, ['var_total_18_19']] = (df.loc[i, ['total_19']].values[0] -
                                                         df.loc[i, ['total_18']].values[0]) / \
                                                        df.loc[i, ['total_18']].values[0]

""" Plots
"""
if 1 == 1:
    """ Plot yearly passengers data
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['domestic'],
                                 name='Domestic'))
        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['international'],
                                 name='International'))
        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['total'],
                                 name='Total'))
        fig.update_xaxes(
            ticktext=["2016", "2017", "2018", "2019"],
            tickvals=df_passengers_total.index,
        )
        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Yearly Passengers (2016-2019)',
                          xaxis_title='Time (years)',
                          yaxis_title='Total Passengers',
                          showlegend=True)

        fig.show()

    """ Plot flights yearly
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['domestic'],
                                 name='Domestic'))
        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['international'],
                                 name='International'))
        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['total'],
                                 name='Total'))
        fig.update_xaxes(
            ticktext=["2016", "2017", "2018", "2019"],
            tickvals=df_flights_total.index,
        )
        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Yearly Passengers (2016-2019)',
                          xaxis_title='Time (years)',
                          yaxis_title='Total Passengers',
                          showlegend=True)

    """ Plot monthly passengers 
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_passengers.index,
                                 y=df_passengers['domestic'],
                                 name='Domestic',
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=df_passengers.index,
                                 y=df_passengers['international'],
                                 name='International',
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=df_passengers.index,
                                 y=df_passengers['total'],
                                 name='Total',
                                 line=dict(color='royalblue',
                                           width=3
                                           )
                                 ))
        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Monthly Passengers (2016-2019)',
                          xaxis_title='Time (Months)',
                          yaxis_title='Total Passengers',
                          showlegend=True)

        fig.show()

    """ Plot monthly flights 
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_flights.index,
                                 y=df_flights['domestic'],
                                 name='Domestic',
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=df_flights.index,
                                 y=df_flights['international'],
                                 name='International',
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=df_flights.index,
                                 y=df_flights['total'],
                                 name='Total',
                                 line=dict(color='firebrick',
                                           width=3
                                           )
                                 ))

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Monthly Flights (2016-2019)',
                          xaxis_title='Time (Months)',
                          yaxis_title='Total Flights',
                          showlegend=True)

        fig.show()

    """ Yearly flights and passengers
    """
    if 1 == 1:
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['domestic'],
                                 name='Passengers - Domestic',
                                 line=dict(color='royalblue',
                                           width=4,
                                           dash='dash'
                                           )
                                 ), secondary_y=False
                      )
        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['international'],
                                 name='Passengers - International',
                                 line=dict(color='royalblue',
                                           width=4,
                                           dash='dot'
                                           )
                                 ), secondary_y=False
                      )
        fig.add_trace(go.Scatter(x=df_passengers_total.index,
                                 y=df_passengers_total['total'],
                                 name='Passengers - Total',
                                 line=dict(color='royalblue', width=4,
                                           )
                                 ), secondary_y=False
                      )

        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['domestic'],
                                 name='Flights - Domestic',
                                 line=dict(color='firebrick', width=4,
                                           dash='dash'
                                           )
                                 ), secondary_y=True
                      )
        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['international'],
                                 name='Flights - International',
                                 line=dict(color='firebrick', width=4,
                                           dash='dot'
                                           )
                                 ), secondary_y=True
                      )
        fig.add_trace(go.Scatter(x=df_flights_total.index,
                                 y=df_flights_total['total'],
                                 name='Flights - Total',
                                 line=dict(color='firebrick', width=4,
                                           )
                                 ), secondary_y=True
                      )

        fig.update_xaxes(
            ticktext=["2016", "2017", "2018", "2019"],
            tickvals=df_flights_total.index,
        )

        fig.update_yaxes(title_text="<b>Passengers</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Flights</b>", secondary_y=True)

        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - (2016-2019)',
                          xaxis_title='Time (years)',
                          showlegend=True)

        fig.update_layout(legend=dict(x=-.1, y=-0.7))

        fig.show()

    """ Plot monthly passengers (together)
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['domestic']['2018-01-31':'2019-01-31'].values,
                                 name='2018-Domestic',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['domestic']['2019-01-31':'2020-12-31'].values,
                                 name='2019-Domestic',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['total']['2018-01-31':'2019-01-31'].values,
                                 name='2018-Total',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['total']['2019-01-31':'2020-12-31'].values,
                                 name='2019-Total',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3
                                           )
                                 ))

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['international']['2018-01-31':'2019-01-31'].values,
                                 name='2018-International',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_passengers['international']['2019-01-31':'2020-12-31'].values,
                                 name='2019-International',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))

        fig.update_xaxes(
            ticktext=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                      "November", "December"],
            tickvals=np.arange(0, 12),
        )
        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Monthly Passengers (2018-2019)',
                          xaxis_title='Time (Months)',
                          yaxis_title='Monthly Passengers',
                          showlegend=True)

        fig.show()

    """ Plot monthly flights (together)
    """
    if 1 == 1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['domestic']['2018-01-31':'2019-01-31'].values,
                                 name='2018-Domestic',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['domestic']['2019-01-31':'2020-12-31'].values,
                                 name='2019-Domestic',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dot'
                                           )
                                 ))

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['total']['2018-01-31':'2019-01-31'].values,
                                 name='2018-Total',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['total']['2019-01-31':'2020-12-31'].values,
                                 name='2019-Total',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3
                                           )
                                 ))

        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['international']['2018-01-31':'2019-01-31'].values,
                                 name='2018-International',
                                 legendgroup="2018",
                                 line=dict(color='firebrick',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))
        fig.add_trace(go.Scatter(x=np.arange(0, 12),
                                 y=df_flights['international']['2019-01-31':'2020-12-31'].values,
                                 name='2019-International',
                                 legendgroup="2019",
                                 line=dict(color='royalblue',
                                           width=3,
                                           dash='dash'
                                           )
                                 ))

        fig.update_xaxes(
            ticktext=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                      "November", "December"],
            tickvals=np.arange(0, 12),
        )
        fig.update_layout(showlegend=True)
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Athens Airport - Monthly Flights (2018-2019)',
                          xaxis_title='Time (Months)',
                          yaxis_title='Monthly Flights',
                          showlegend=True)

        fig.show()

if 1 == 1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_passengers_var.index,
                             y=df_passengers_var['var_domestic_18_19'],
                             name='Domestic',
                             line=dict(color='firebrick',
                                       width=3,
                                       dash='dot'
                                       )
                             ))
    fig.add_trace(go.Scatter(x=df_passengers_var.index,
                             y=df_passengers_var['var_international_18_19'],
                             name='International',
                             line=dict(color='royalblue',
                                       width=3,
                                       dash='dot'
                                       )
                             ))

    fig.add_trace(go.Scatter(x=df_passengers_var.index,
                             y=df_passengers_var['var_total_18_19'],
                             name='Total',
                             line=dict(color='firebrick',
                                       width=3,
                                       )
                             ))

    fig.update_xaxes(
        ticktext=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"],
        tickvals=np.arange(0, 12),
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
    fig.update_layout(title='Athens Airport - Passengers Variation 19-18',
                      xaxis_title='Time (Months)',
                      yaxis_title='Passengers Variation (%)',
                      showlegend=True)

    fig.show()
