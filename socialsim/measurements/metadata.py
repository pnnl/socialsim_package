import pandas as pd
import numpy  as np
import sys

import datetime

class MetaData:
    def __init__(self, content_data=False, user_data=False, verbose=True, 
        community_directory=None, node_file=None):
        """
        Description:

        Input:

        Output:
        
        """
        if node_file is None:
            self.node_list = 'all'
        else:
            self.node_list = self.load_node_list(node_file)

        if community_directory is None:
            pass
        else:
            self.community_directory = community_directory

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

    def build_communities(self, content_data, user_data):
        """
        Description:

        Input:

        Output:

        """

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
        dataset.columns = ['content', 'created_at', 'owner_id', 'language']
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