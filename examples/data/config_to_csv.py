import json
import pandas as pd

import pprint

def config_to_df(config):

    platforms = []
    meas_types = []
    scales = []
    measurement_names = []
    measurement_functions = []
    meas_args = []
    metric_names = []
    metric_funcs = []
    metric_args = []
    temporals = []

    for platform in config.keys():
        for meas_type in config[platform]:
            for scale in config[platform][meas_type]:
                for meas_name in config[platform][meas_type][scale]:
                    
                    meas = config[platform][meas_type][scale][meas_name] 

                    if 'measurement' in config[platform][meas_type][scale][meas_name].keys():

                        for metric_name in meas['metrics'].keys():

                            measurement_functions.append(meas['measurement'])
                            metric_names.append(metric_name)
                            metric_funcs.append(meas['metrics'][metric_name]['metric'])
                            platforms.append(platform)
                            meas_types.append(meas_type)
                            scales.append(scale)
                            measurement_names.append(meas_name)

                            if 'metric_args' in meas['metrics'][metric_name]:
                                metric_args.append(meas['metrics'][metric_name]['metric_args'])
                            else:
                                metric_args.append({})

                            if 'temporal_vs_batch' in meas.keys():
                                temporals.append(meas['temporal_vs_batch'])
                            else:
                                temporals.append('Batch')

                            if 'measurement_args' in meas.keys():
                                meas_args.append(meas['measurement_args'])
                            else:
                                meas_args.append({})                        
                    else:
                        print('No measurement:',meas_name)

    df = pd.DataFrame({'Platform':platforms,
                       'Measurement Type':meas_types,
                       'Group Scale': scales,
                       'Measurement': measurement_names,
                       'Measurement Function': measurement_functions,
                       'Measurement Args': meas_args,
                       'Metrics': metric_names,
                       'Metric Functions':metric_funcs,
                       'Metric Args':metric_args,
                       'Temporal':temporals})


    return(df)


def main():

    fn = 'cp3_s1_configuration.json'

    with open(fn,'r') as f:
        config = json.load(f)

        
    df = config_to_df(config)
    
    df.to_csv('cp3_s1_configuration.csv',index=False)


if __name__ == "__main__":
    main()
