import socialsim as ss
import json
import os
import sys


def load_configuration(filepath):

    with open(filepath) as f:
        configuration = json.load(f)

    return configuration

def overview(d, n=0, max_n=5):
    if n==max_n:
        return None

    spacer = '   |'

    for i in d.keys():
        print(spacer*n+'--'+i)

        if type(d[i]) is str:
            print(spacer*(n+1)+'--'+d[i])
        elif i=='error':
            print(spacer*(n+1)+'--'+type(d[i]).__name__+' : '+str(d[i]))
        else:
            overview(d[i], n=n+1)

    return None     

if __name__=='__main__':
    # Define location of needed files
    config_directory         = './configs/new/'
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
                print('Configuration filename: '+config_filename)

                configuration = {platform: configuration}

                
                for platform in configuration.keys():
                    # print(platform)
                    for measurement_type in configuration[platform].keys():
                        # print('|----'+measurement_type)
                        for scale in configuration[platform][measurement_type].keys():
                            # print('|      |----'+scale)
                            for measurement in configuration[platform][measurement_type][scale].keys():
                                # print('|      |      |----'+measurement)
                                pass

                ground_truth_file_directory = ground_truth_directory+'/'+'scenario'+scenario+'/'+domain+'/'+platform+'/'

                for filepath in os.listdir(ground_truth_file_directory):
                    print('Ground truth file: '+ground_truth_file_directory+filepath)
                    ground_truth = ss.load_data(ground_truth_file_directory+filepath, short=True)

                metadata = None
                task_runner = ss.TaskRunner(ground_truth, metadata, configuration)

                print('-'*30)

                for filepath in os.listdir(submission_directory):
                    with open(submission_directory+filepath) as f:
                        first_line = f.readline()
                        metadata = json.loads(first_line)
   
                        if metadata['scenario']==scenario:
                            pass
                        else:
                            continue

                        if metadata['domain']==domain:
                            pass
                        else:
                            continue

                        if metadata['platform']==platform:
                            pass
                        else:
                            continue

                        print('loading submission with metadata:')
                        print(metadata)

                    dataset = ss.load_data(submission_directory+filepath, short=True)

                    print('Running task runner.')
                    results, logs = task_runner.run(dataset)
                    print('Task complete.')
 
                    logs = logs[0]
                    
                    overview(logs)
                    
                    results = results[0]

                    try:
                       print(type(results['reddit']['cascade']['community']['community_max_depth_distribution']))
                       print(results['reddit']['cascade']['community']['community_max_depth_distribution'])
                    except Exception as error:
                        print(error)
                   



























