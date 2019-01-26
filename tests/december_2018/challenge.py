import socialsim as ss
import json
import os

def load_configuration(filepath):

    with open(filepath) as f:
        configuration = json.load(f)

    return configuration

if __name__=='__main__':
    # Define location of needed files
    config_directory         = './configs/'
    submission_directory     = '/data/socialsim/december_2018/processed_submissions/'
    challenge_data_directory = '/data/socialsim/december_2018/'
    ground_truth_directory   = '/data/socialsim/december_2018/processed_challenge/ground_truth_data'

    # Define scope for scenarios, domains and platforms
    scenarios = ['1'] #, '2']
    domains   = ['cyber', 'cve', 'crypto']
    platforms = ['github', 'reddit', 'twitter']

    # Specify which combination of (scenario, domain, platform) is going to run
    for scenario in scenarios:
        for domain in domains:
            for platform in platforms:

                # Load configuration file
                config_filename = str(scenario)+'_'+platform+'_'+domain+'_config.json'
                configuration = load_configuration(config_directory+config_filename)

                print('-'*80)
                print(config_filename)

                configuration = {platform: configuration}

                """
                for platform in configuration.keys():
                    print(platform)
                    for measurement_type in configuration[platform].keys():
                        print('|----'+measurement_type)
                        for scale in configuration[platform][measurement_type].keys():
                            print('|      |----'+scale)

                            scale_level_dict = {}
                            for measurement in configuration[platform][measurement_type][scale]:
                                scale_level_dict.update(measurement)

                            configuration[platform][measurement_type][scale] = scale_level_dict

                            for measurement in configuration[platform][measurement_type][scale].keys():
                                print('|      |      |----'+measurement)
                """

                ground_truth_file_directory = ground_truth_directory+'/'+'scenario'+scenario+'/'+domain+'/'+platform+'/'

                for filepath in os.listdir(ground_truth_file_directory):
                    print(ground_truth_file_directory+filepath)
                    ground_truth = ss.load_data(ground_truth_file_directory+filepath)

                metadata = None
                task_runner = ss.TaskRunner(ground_truth, metadata, configuration)
