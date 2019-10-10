import pandas as pd
import numpy  as np
import sys
import os
import glob
import joblib
import datetime

class MetaData:
    def __init__(self, content_data=False, user_data=False, info_data=False,
                 community_directory=None, node_file=None, verbose=True,
                 content_node_ids='all', user_node_ids=[], do_build_communities=False,
                 communities=None):

        """
        Description:

        Input:

        Output:
        
        """

        self.do_build_communities = do_build_communities
        
        self.content_node_ids  = content_node_ids
        self.user_node_ids     = user_node_ids

        self.verbose = verbose

        if node_file is None:
            self.node_list = 'all'
        else:
            self.node_list = self.load_node_list(node_file)

        self.community_directory = community_directory            
        self.communities = communities
        if not community_directory is None:
            self.read_communities()

        if content_data!=False:
            self.use_content_data = True

            if verbose: 
                print('Loading content metadata... ', end='', flush=True)

            self.content_data = pd.read_csv(content_data)

            if verbose: 
                print('Done.', flush=True)

            self.content_data = self.preprocessContentMeta(self.content_data)
        else:
            self.use_content_data = False

        if user_data != False:
            self.use_user_data = True

            if verbose: print('Loading user metadata... ', end='', flush=True)
            self.user_data = pd.read_csv(user_data)
            if verbose: print('Done.', flush=True)

            self.user_data = self.preprocessUserMeta(self.user_data)
        else:
            self.use_user_data = False

        if info_data != False:
            self.use_info_data = True
 
            if verbose: print('Loading info metadata... ', end='', flush=True)
            self.info_data = pd.read_csv(info_data)
            if verbose: print('Done.', flush=True)
        else:
            self.use_info_data = False

        model_filepath = os.path.dirname(__file__)
        model_filepath = model_filepath+'/model_parameters/best_model.pkl'

        self.estimator = joblib.load(model_filepath)

    def read_communities(self):

        community_fns = glob.glob(self.community_directory + '/*')
        
        self.communities = {}
        for fn in community_fns:
            
            comm_id = fn.split('/')[-1].split('.')[0]
            comm_members = []
            with open(fn,'r') as f:
                comm_members = [line.rstrip('\n') for line in f]

            if len(comm_members) > 0:
                self.communities[comm_id] = comm_members

            if self.verbose: 
                print(self.communities)

    def build_communities(self, content_data, user_data):
        """
        Description:

        Input:

        Output:

        """
        
        if not self.do_build_communities:
        	return {}
          
        print('Building communities... ', end='')

        communities = {}

        repo_communities = ['language']

        user_communities = ['country', 'company']


        for community_type in repo_communities:
            communities.update({community_type:{}})

            a = content_data[community_type].unique()
            unique_communities = np.random.choice(a, size=100, replace=False)

            for community in unique_communities:
                subset = content_data[content_data[community_type]==community]
                community_list = subset['content'].tolist()
                communities[community_type].update({community:community_list})

        for community_type in user_communities:
            communities.update({community_type:{}})

            a = user_data[community_type].unique()
            unique_communities = np.random.choice(a, size=100, replace=False)

            for community in unique_communities:
                subset = user_data[user_data[community_type]==community]
                community_list = subset['user'].tolist()
                communities[community_type].update({community:community_list})

        print('Done.')

        return communities


    def preprocessContentMeta(self, dataset):
        if len(dataset.columns) == 4:
            dataset.columns = ['content', 'created_at', 'owner_id', 'language']
        elif len(dataset.columns) == 3:
            dataset.columns = ['created_at','owner_id','content']
        dataset['created_at'] = pd.to_datetime(dataset['created_at'])
        return dataset


    def preprocessUserMeta(self, dataset):
        try:
            dataset.columns = ['user','created_at','location','company']
        except:
            dataset.columns = ['user','created_at','city','country','company']

        dataset['created_at'] = pd.to_datetime(dataset['created_at'])
        return dataset


    def load_node_list(self, node_file):
        """
        Description: 

        Input:
            :node_file

        Output:
            :node_list:
        """

        with open(node_file) as f:
            node_list = f.read().splitlines()

        return node_list
