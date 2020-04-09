import pandas as pd
import numpy as np
import json
import collections
import zipfile
from .definitions import *
import scipy.stats

## ---------------------------------------------------------------------------------------------------------------------
## Loading related functions
## ---------------------------------------------------------------------------------------------------------------------

DEFAULT_SEPARATOR = '-'

def try_to_float(s):
    try:
        return float(s)
    except:
        return np.NaN


def flatten(d, parent_key='', sep=DEFAULT_SEPARATOR):
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

def split_measurement_and_args(meas_str, sub_types=['_sliding_','_expanding_']): 
        for sub_type in sub_types:
            if sub_type in meas_str:
                splitvals = meas_str.split(sub_type)
                meas, args = splitvals[0], sub_type+sub_type.join(splitvals[1:])
                return {'measurement':meas, 'measurement_args':args}
        return {'measurement':meas_str, 'measurement_args':''}
    
def format_from_metrics_json_df(metrics_jsons_df, sep=DEFAULT_SEPARATOR):
    # extract measurement columns from data loaded
    non_meas_cols = ['eval_ver', 'identifier', 'scenario', 'team', 'team_alias', 'tira_dataset', 'domain', 'fn']
    meas_cols = []
    for col in metrics_jsons_df.columns:
        if sep in col:
            platform = col.split(sep)[0]
            if platform in ['multi_platform', 'twitter', 'github', 'reddit', 'telegram', 'youtube']:
                meas_cols.append(col)

    if 'domain' not in metrics_jsons_df.columns:
        metrics_jsons_df['domain'] = ''

    # reformat from wide to long dataframe
    wide_df = metrics_jsons_df.copy()
    df = pd.melt(wide_df, id_vars=non_meas_cols, value_vars=meas_cols)
    # extract platform, measurement type (measurement_type), scale, measurement, metric, and node (e.g. information ID)
    # from 'variable' column
    df['platform'] = df['variable'].apply(lambda x: x.split(sep)[0])
    df['measurement_type'] = df['variable'].apply(lambda x: x.split(sep)[1])
    df['scale'] = df['variable'].apply(lambda x: x.split(sep)[2])
    df['measurement'] = df['variable'].apply(lambda x: x.split(sep)[3])
    
    df['measurement_args'] = df['measurement'].apply(lambda x: split_measurement_and_args(x)['measurement_args'])
    df['measurement'] = df['measurement'].apply(lambda x: split_measurement_and_args(x)['measurement'])
           
    df['metric'] = df['variable'].apply(lambda x: x.split(sep)[4] if len(x.split(sep)) >= 5 else '')
    df['node'] = df['variable'].apply(lambda x: x.split(sep)[5] if len(x.split(sep)) >= 6 else '')
    # remove values that contain 'failed to run.'
    df = df[['failed to run.' not in str(x) for x in  df['value']]].copy()
    # force 'value' to float type
    df['value'] = df['value'].astype(float)
    # format metrics results data with categories
    df = format_metrics_results_data_with_categories(df).dropna(subset=['value'])
    return df


def read_metrics_files_to_dataframe(filepaths, sep=DEFAULT_SEPARATOR):
    metrics = []
    for fpath in filepaths:
        m = json.load(open(fpath, 'r'))
        m['fn'] = fpath
        metrics.append(flatten(m, sep=sep))
    df = pd.DataFrame(metrics)  
    df['scenario'] = df['scenario'].astype(str)
    return df


def load_results_from_metrics_files(metrics_json_filepaths, sep=DEFAULT_SEPARATOR):
    # read metrics json to a combined dataframe
    metrics_jsons_df = read_metrics_files_to_dataframe(metrics_json_filepaths, sep=sep)
    # format metrics df to metrics results data with categories
    df = format_from_metrics_json_df(metrics_jsons_df, sep=sep)
    return df


def read_metrics_files_to_dataframe_from_zipped(zipfile_with_metrics_files, sep=DEFAULT_SEPARATOR):
    metrics = []

    zfile = zipfile.ZipFile(f'./{zipfile_with_metrics_files}')
    fnames = zfile.namelist()
    metrics_fnames = [x for x in fnames if '_metrics.json' in x]

    for metrics_fname in metrics_fnames:
        m = json.load(zfile.open(metrics_fname))
        metrics.append(flatten(m, sep=sep))

    metrics_jsons_df = pd.DataFrame(metrics)
    metrics_jsons_df['scenario'] = metrics_jsons_df['scenario'].astype(str)

    return metrics_jsons_df


def load_results_from_metrics_files_in_zipfile(zipfile_with_metrics_files, sep=DEFAULT_SEPARATOR):
    # read metrics json to a combined dataframe
    metrics_jsons_df = read_metrics_files_to_dataframe_from_zipped(zipfile_with_metrics_files, sep=sep)
    # format metrics df to metrics results data with categories
    df = format_from_metrics_json_df(metrics_jsons_df, sep=sep)
    return df



def rename_identifier_if_mapped(identifier, model_renaming_mapping):
    if identifier in model_renaming_mapping.keys():
        return model_renaming_mapping[identifier]
    else:
        return identifier

def confidence_interval_width(data, confidence=0.95):
    data = [x for x in data if x != np.nan]
    if len(data) < 2: return np.nan
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n- 1)
    return h


def get_metrics_averages(df):
    """
    Function to averaged metrics results to get mean value per metric
    :param df: pandas dataframe representation of _metrics.json files
    :return: results dataframe after metric performance is averaged across multiple submissions of the same model
    """
    df.scenario = df.scenario.astype(int).astype(str)

    base = df.copy()
    base.loc[:,'value_CI'] = base['value'] 
    gb_cols = list(set(base.columns)-set(['value','value_CI','fn'])) 
    base = base.groupby(gb_cols, as_index=False).agg({'value': np.mean, 
                                                     'value_CI': lambda x: confidence_interval_width(x)})  
    return base



def load_averaged_model_metrics_json(list_of_metrics_filepaths, model_renaming_mapping={}): 
    df = load_results_from_metrics_files(list_of_metrics_filepaths, sep=' <> ') 
    if model_renaming_mapping!={}:
        df['identifier'] = df['identifier'].apply(lambda identifier: 
                                                  rename_identifier_if_mapped(identifier, model_renaming_mapping)) 
         
    return get_metrics_averages(df).drop(columns=['team_alias','variable','eval_ver','domain'])
 


