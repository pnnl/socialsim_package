import sys
import os.path

socialsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                              os.path.pardir))

sys.path.append(socialsim_path)

import socialsim as ss


data_directory = 'test_datasets\\reddit\\'

filename = 'reddit_data_cyber_20180101_to_20180107_1week.csv'

ss.csv_to_json(data_directory+filename, 'PNNL', '1', 'cyber', 'reddit')
