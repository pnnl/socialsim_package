import pandas as pd
import json

import ast

import pprint


def df_to_config(df):

    platform_grouped = df.groupby('Platform')

    output = {}

    for platform, df_platform in platform_grouped:
        
        meas_type_grouped = df_platform.groupby('Measurement Type')

        output[platform] = {}

        for meas_type, df_meas_type in meas_type_grouped:

            scale_grouped = df_meas_type.groupby('Group Scale')
            
            output[platform][meas_type] = {}

            for scale, df_scale in scale_grouped:

                measurement_grouped = df_scale.groupby('Measurement')

                output[platform][meas_type][scale] = {}

                for meas_name, df_meas in measurement_grouped:

                    print(meas_name)

                    args = ast.literal_eval(df_meas['Measurement Args'].unique()[0])

                    meas_dict = {'measurement':df_meas['Measurement Function'].unique()[0],
                                 'measurement_args':args,
                                 'scale':scale,
                                 'temporal_or_batch':df_meas['Temporal'].unique()[0],
                                 'metrics':{}
                                 }
                                 
                    for i, row in df_meas.iterrows():

                        print(row['Metrics'])

                        args = ast.literal_eval(row['Metric Args'])

                        meas_dict['metrics'][row['Metrics']] = {'metric':row['Metric Functions'],
                                                                'metric_args':args}
                    

                    pprint.pprint(meas_dict)

                    output[platform][meas_type][scale][meas_name] = meas_dict
        
            
    pprint.pprint(output)
        
    return(output)

def main():

    df = pd.read_csv('cp3_s1_configuration.csv')
    
    config = df_to_config(df)
    
    with open('cp3_s1_configuration.json','w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()
