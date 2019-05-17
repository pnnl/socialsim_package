from .measurements import MeasurementsBaseClass
from .recurrence import BurstDetection
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import community
from collections import defaultdict
import pysal
import warnings

# community detection algorithms
# More algorithms: https://networkx.github.io/documentation/stable/reference/algorithms/community.html

def louvain_method(g):
    '''
    https://python-louvain.readthedocs.io/en/latest/
    Fast unfolding of communities in large networks, Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Renaud Lefebvre, Journal of Statistical Mechanics: Theory and Experiment 2008(10), P10008 (12pp)
    :param g: networkx Graph
    '''
    groups = defaultdict(list)
    for uid, group in community.best_partition(g).items():
        groups[group].append(uid)
    return list(groups.values())


class PersistentGroupsMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset_df, configuration={}, metadata=None, id_col='nodeID', timestamp_col="nodeTime", userid_col="nodeUserID", platform_col="platform", content_col="informationID", log_file='group_formation_measurements_log.txt', selected_content=None, time_granularity='12H', parentid_col='parentID', community_detection_algorithm=louvain_method):
        """
        :param dataset_df: dataframe containing all posts for all communities (Eg. coins for scenario 2) in all platforms
        :param timestamp_col: name of the column containing the time of the post
        :param id_col: name of the column containing the post id
        :param userid_col: name of the column containing the user id
        :param content_col: name of the column containing the content the simulation was done for eg. coin name
        :param community_detection_algorithm: a function that takes a networkx Graph as in input and returns a list of list i.e. [[users_in_group1], [users_in_group2], ...}
        """
        super(PersistentGroupsMeasurements, self).__init__(dataset_df, configuration, log_file=log_file)
        self.dataset_df          = dataset_df
        self.timestamp_col       = timestamp_col
        self.id_col              = id_col
        self.userid_col          = userid_col
        self.platform_col        = platform_col
        self.content_col = content_col
        self.measurement_type    = 'persistent_groups'
        self.metadata            = metadata 
        self.selected_content = selected_content if selected_content is not None else self.metadata.node_list
        if self.selected_content == 'all':
            self.selected_content = None

        self.min_date = self.dataset_df[self.timestamp_col].min()
        self.max_date = self.dataset_df[self.timestamp_col].max()
            
        self.gammas = {k:None for k in self.dataset_df[self.content_col].unique()}

        if not self.metadata is None:
            if self.metadata.use_info_data and 'gamma' in self.metadata.info_data.columns:
                self.gammas.update(self.metadata.info_data[[self.content_col,'gamma']].set_index(self.content_col).to_dict()['gamma'])

        self.time_granularity = time_granularity
        self.parentid_col = parentid_col
        self.community_detection_algorithm = community_detection_algorithm
        self.get_network_from_bursts()

    def get_network_from_bursts(self, bursts_count_threshold=1, user_interaction_weight_threshold=1):        
        """
        get bursts in activity for units of information, get network of connected users that parcipate in these bursts
        :param bursts_count_threshold: threshold for the number of bursts in activity for an information to be considered
        :param user_interaction_weight_threshold: threshold for the number of burts two users must participate in together in order for them to be connected
        """ 
        def get_burst_user_connections_df(content_id, content_df, burst_interval, userid_col='from_id', timestamp_col='date'):
            '''connect users who participate in the same burst'''
            burst_df = content_df[(content_df[self.timestamp_col].between(burst_interval[0], burst_interval[1], inclusive=True)) & (~content_df[self.userid_col].isna())].copy()
            uids = list(sorted(burst_df[self.userid_col].unique())) 
            if len(uids) < 2:
                return []            
            content_user_connections = []
            for i, uid1 in enumerate(uids):
                for uid2 in uids[i+1:]:
                    content_user_connections.append({'content': content_id,
                                    'uid1': uid1,
                                    'uid2': uid2,
                                    'weight': 1})
            return content_user_connections

        user_connections = []
        count = 0
        n_ids = self.dataset_df[self.content_col].nunique()
        for content_id, content_df in self.dataset_df.groupby(self.content_col):
            if self.selected_content is not None and content_id not in self.selected_content:
                continue
            count += 1
            burstDetection = BurstDetection(dataset_df=content_df, metadata=self.metadata, id_col=self.id_col,
                                            timestamp_col=self.timestamp_col, platform_col=self.platform_col, 
                                            time_granularity=self.time_granularity,
                                            min_date=self.min_date, max_date=self.max_date)
            burst_intervals = burstDetection.detect_bursts(self.gammas[content_id])
            if len(burst_intervals) < bursts_count_threshold:
                continue
            for burst_interval in burst_intervals:
                user_connections.extend(get_burst_user_connections_df(content_id, content_df, burst_interval))
            
        user_network_df = pd.DataFrame(user_connections)
        if 'uid1' not in user_network_df.columns:
            warnings.warn("No bursts detected in any information IDs. Persistent group measurements cannot be run. They will fail with uid1 KeyError.")
        user_network_df = user_network_df.groupby(['uid1', 'uid2'])['weight'].sum().reset_index()

        self.user_network_df = user_network_df[user_network_df['weight']>=user_interaction_weight_threshold]
        user_interaction_nx = nx.from_pandas_edgelist(user_network_df,
                                    'uid1', 'uid2', 'weight')
        print('num nodes: ', user_interaction_nx.number_of_nodes(), 'num edges: ', user_interaction_nx.number_of_edges())
        self.groups = self.community_detection_algorithm(user_interaction_nx)
        print('Number of groups: ', len(self.groups))

    def number_of_groups(self):	
        '''How many different clusters of users are there?'''
        return len(self.groups)
        
    def group_size_distribution(self):
        '''How large are the groups of users?'''	
        return [len(group_users) for group_users in self.groups]
        
    def distribution_of_content_discussion_over_groups(self):	
        '''Do groups focus on individual information IDs or a larger set of info IDs?'''	

        content_ids = self.dataset_df.groupby(self.content_col)[self.id_col].count().reset_index()
        content_ids.columns = [self.content_col,'total_value']

        meas = []
        for i, group_users in enumerate(self.groups):
            group_df = self.dataset_df[self.dataset_df[self.userid_col].isin(group_users)]

            info_id_counts = group_df.groupby(self.content_col)[self.id_col].count().reset_index()
            info_id_counts.columns = [self.content_col,'value']
            info_id_counts = info_id_counts.merge(content_ids,on=self.content_col,how='right').fillna(0)

            info_id_counts = list(info_id_counts['value'].values)

            #inequality among information IDs within the group
            content_gini = pysal.explore.inequality.gini.Gini(info_id_counts).g

            meas.append(content_gini)

        return meas

    def internal_versus_external_interaction_rates(self):	
        '''How much do group members interact with each other versus non-group members?'''
        internal_links = 0
        external_links = 0
        for i, group_users in enumerate(self.groups):
            internal_links_df = self.user_network_df[(self.user_network_df['uid1'].isin(group_users)) & (self.user_network_df['uid2'].isin(group_users))]
            all_links_df = self.user_network_df[(self.user_network_df['uid1'].isin(group_users)) | (self.user_network_df['uid2'].isin(group_users))]  # all links made by users in that group
            internal_links += sum(internal_links_df['weight'].values)
            external_links += sum(all_links_df['weight'].values) - sum(internal_links_df['weight'].values)
        return external_links / internal_links

    def group_versus_total_volume_of_activity(self,time_granularity=None):	
        '''How much does the most prolific group dominate the discussion of a particular info ID over time?'''		

        if time_granularity is None:
            time_granularity = self.time_granularity

        dataset_counts_df = self.dataset_df.set_index(self.timestamp_col).\
            groupby([pd.Grouper(freq=time_granularity), self.content_col]).size().reset_index(name='total_activity')
        
        group_content_timeseries = {}       
        for content_id, content_df in self.dataset_df.groupby(self.content_col):

            #users in the group with the most posts related to this content ID
            prolific_group_users = self.groups[np.argmax(np.array([len(content_df[content_df[self.userid_col].isin(group_users)]) for group_users in self.groups]))]

            group_df = content_df[content_df[self.userid_col].isin(prolific_group_users)]

            group_counts_df = group_df.set_index(self.timestamp_col).\
                groupby([pd.Grouper(freq=time_granularity), self.content_col]).size().reset_index(name='group_activity')

            merged_df = dataset_counts_df.merge(group_counts_df, how='outer', on=[self.content_col, self.timestamp_col])
            merged_df.fillna(0, inplace=True)
            merged_df['value'] = merged_df['group_activity'] / merged_df['total_activity']

            group_content_timeseries[content_id] = merged_df.drop(columns=[self.content_col, 'total_activity', 'group_activity'])
        return group_content_timeseries

    def seed_post_versus_response_actions_ratio(self):  
        '''How much does the group seed new content?'''
        group_seed_post_ratio = []
        for i, group_users in enumerate(self.groups):
            group_df = self.dataset_df[self.dataset_df[self.userid_col].isin(group_users)]
            idx = (group_df[self.parentid_col] == group_df[self.id_col]) | (group_df['actionType'].isin(['CreateEvent','IssuesEvent','PullRequestEvent']))
            group_seed_post_ratio.append(len(group_df[idx]) / float(len(group_df))) 
        return group_seed_post_ratio       
        
