import pandas as pd
import json
import ast
from config_to_csv import config_to_df
from csv_to_config import df_to_config


def add_date_limits(meas_dict,date_min,date_max):
    meas_dict['date_range'] = [date_min,date_max]
    return(meas_dict)

def add_windows_to_config(config, first_month, last_month):

    config = config_to_df(config)

    meas_to_include = ['number_of_shares',
                       'number_of_shares_over_time',
                       'distribution_of_shares',
                       'order_of_spread',
                       'time_delta',
                       'overlapping_users']


    last_month = str(pd.to_datetime(last_month) + pd.DateOffset(months=1))[:7] 

    dates = [str(x)[:7] + '-01' for x in pd.date_range(first_month,last_month,freq='M')]


    configs = []

    for i, date in enumerate(dates[1:]):


        sliding_config = config[config['Measurement Function'].isin(meas_to_include)].copy()

        sliding_config['Measurement Args'] = sliding_config['Measurement Args'].apply(lambda x: add_date_limits(x,dates[i],date))
        sliding_config['Measurement'] = sliding_config['Measurement'].apply(lambda x: x + '_sliding_' + dates[i] + '_to_' + date)

        configs.append(sliding_config)


        expanding_config = config[config['Measurement Function'].isin(meas_to_include)].copy()
        expanding_config['Measurement Args'] = expanding_config['Measurement Args'].apply(lambda x: add_date_limits(x,dates[0],date))
        expanding_config['Measurement'] = expanding_config['Measurement'].apply(lambda x: x + '_expanding_' + dates[0] + '_to_' + date)

        configs.append(expanding_config)


    df = pd.concat([config] + configs)

    for col in df.columns:
        if 'arg' in col.lower():
            df[col] = df[col].astype(str)

    config = df_to_config(df)

    return(config)


def main():

    scenario = 's2'
    first_month = '2018-06'
    last_month = '2019-01'

    fn = 'cp3_' + scenario + '_configuration.json'

    with open(fn, 'r') as f:
        config = json.load(f)

    config = add_windows_to_config(config, first_month, last_month)

    with open('cp3_' + scenario + '_windows_configuration.json','w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
