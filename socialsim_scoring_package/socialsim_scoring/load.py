import pandas as pd
import numpy as np
import json
import collections
import zipfile
from .definitions import *


## ---------------------------------------------------------------------------------------------------------------------
## Loading related functions
## ---------------------------------------------------------------------------------------------------------------------

def try_to_float(s):
    try:
        return float(s)
    except:
        return np.NaN


def flatten(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def format_metrics_results_data_with_categories(df, cp3=True):
    df_0 = df.copy()
    s1_cats = S1_MEASUREMENT_CATEGORIES.assign(scenario=1)

    s2_cats = S2_MEASUREMENT_CATEGORIES.assign(scenario=2)

    s_cats = pd.concat((s1_cats, s2_cats))

    s_cats['scenario'] = s_cats['scenario'].astype(int).astype(str)
    df_0['scenario'] = df_0['scenario'].astype(int).astype(str)
    if cp3:
        # convert measurement_type to the new class structure
        # (scale_of_spread, cross-platform_spread, social_network_os_spread, structure_of_spread)
        df_0['measurement_type'] = [meas_type_to_class[x] for x in df_0['measurement_type']]

    on = ['scenario', 'measurement_type', 'platform', 'scale', 'measurement']
    df = pd.merge(s_cats, df_0, on=on)

    # convert columns to floats or drop
    df.value = df.value.apply(try_to_float)
    df.node.fillna('n/a', inplace=True)
    dfdropna = df.dropna()


    print('Teams represented: {}'.format(sorted(list(set(dfdropna['team'])))))
    print('Scenarios represented: {}'.format(sorted(list(set(dfdropna['scenario'])))))

    return df


def format_from_metrics_json_df(metrics_jsons_df):
    # extract measurement columns from data loaded
    non_meas_cols = ['eval_ver', 'identifier', 'scenario', 'team', 'team_alias', 'tira_dataset', 'domain']
    meas_cols = []
    for col in metrics_jsons_df.columns:
        if '-' in col:
            platform = col.split('-')[0]
            if platform in ['multi_platform', 'twitter', 'github', 'reddit', 'telegram', 'youtube']:
                meas_cols.append(col)

    if 'domain' not in metrics_jsons_df.columns:
        metrics_jsons_df['domain'] = ''

    # reformat from wide to long dataframe
    wide_df = metrics_jsons_df.copy()
    df = pd.melt(wide_df, id_vars=non_meas_cols, value_vars=meas_cols)
    # extract platform, measurement type (measurement_type), scale, measurement, metric, and node (e.g. information ID)
    # from 'variable' column
    df['platform'] = df['variable'].apply(lambda x: x.split('-')[0])
    df['measurement_type'] = df['variable'].apply(lambda x: x.split('-')[1])
    df['scale'] = df['variable'].apply(lambda x: x.split('-')[2])
    df['measurement'] = df['variable'].apply(lambda x: x.split('-')[3])
    df['metric'] = df['variable'].apply(lambda x: x.split('-')[4] if len(x.split('-')) >= 5 else '')
    df['node'] = df['variable'].apply(lambda x: x.split('-')[5] if len(x.split('-')) >= 6 else '')
    # remove values that contain 'failed to run.'
    df = df[['failed to run.' not in str(x) for x in  df['value']]].copy()
    # force 'value' to float type
    df['value'] = df['value'].astype(float)
    # format metrics results data with categories
    df = format_metrics_results_data_with_categories(df)
    return df


def read_metrics_files_to_dataframe(filepaths):
    metrics = []
    for fpath in filepaths:
        m = json.load(open(fpath, 'r'))
        metrics.append(flatten(m))

    df = pd.DataFrame(metrics)
    df['scenario'] = df['scenario'].astype(str)
    return df


def load_results_from_metrics_files(metrics_json_filepaths):
    # read metrics json to a combined dataframe
    metrics_jsons_df = read_metrics_files_to_dataframe(metrics_json_filepaths)
    # format metrics df to metrics results data with categories
    df = format_from_metrics_json_df(metrics_jsons_df)
    return df


def read_metrics_files_to_dataframe_from_zipped(zipfile_with_metrics_files):
    metrics = []

    zfile = zipfile.ZipFile(f'./{zipfile_with_metrics_files}')
    fnames = zfile.namelist()
    metrics_fnames = [x for x in fnames if '_metrics.json' in x]

    for metrics_fname in metrics_fnames:
        m = json.load(zfile.open(metrics_fname))
        metrics.append(flatten(m))

    metrics_jsons_df = pd.DataFrame(metrics)
    metrics_jsons_df['scenario'] = metrics_jsons_df['scenario'].astype(str)

    return metrics_jsons_df


def load_results_from_metrics_files_in_zipfile(zipfile_with_metrics_files):
    # read metrics json to a combined dataframe
    metrics_jsons_df = read_metrics_files_to_dataframe_from_zipped(zipfile_with_metrics_files)
    # format metrics df to metrics results data with categories
    df = format_from_metrics_json_df(metrics_jsons_df)
    return df

