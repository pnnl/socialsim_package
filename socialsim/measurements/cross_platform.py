import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from itertools import combinations
import pprint
import warnings
import re

from .measurements import MeasurementsBaseClass

from ..utils import add_communities_to_dataset
import re

class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration={}, metadata=None, 
        platform_col="platform", timestamp_col="nodeTime", 
        user_col="nodeUserID", content_col="informationID", 
        community_col="community", 
        log_file='cross_platform_measurements_log.txt', 
        node_list=None, community_list=None):
        """
        :param dataset: dataframe containing all pieces of content and associated data, sorted by time
        :param configuration:
        :param platform_col: name of the column containing the platforms
        :param timestamp_col: name of the column containg the time of the content
        :param user_col: name of the column containing the user ids of each piece of content
        :param content_col: name of the column where the content can be found
        :param community_col: name of the column containing the subset of content in a community
        :param log_file:
        """
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, 
            log_file=log_file)

        self.dataset            = dataset
        self.timestamp_col      = timestamp_col
        self.user_col           = user_col
        self.platform_col       = platform_col
        self.content_col        = content_col
        self.community_col      = community_col

        self.measurement_type = 'cross_platform'
        
        if metadata is None:
            self.community_set = self.dataset
            self.community_set[self.community_col] = "Default Community"

            if node_list == "all":
                self.node_list = self.dataset[self.content_col].tolist()
            elif node_list is not None:
                self.node_list = node_list
            else:
                self.node_list = []
        else:

            if (metadata.community_directory is None and metadata.communities is None):
                self.community_set = self.dataset
                self.community_set[self.community_col] = "Default Community"
            else:
                community_directory = metadata.community_directory
                communities = metadata.communities
                self.community_set = add_communities_to_dataset(dataset,
                                                                community_directory,
                                                                communities)
            
            self.node_list = metadata.node_list

        if self.community_set is not None:
            if community_list is not None and len(community_list) > 0:
                self.community_list = community_list
            else:
                self.community_list = self.community_set[self.community_col].dropna().unique()
        else:
            self.community_list = []


    def list_measurements(self):
        count = 0
        for f in dir(self):
            if not f.startswith('_'):
                func = getattr(self, f)
                if callable(func):
                    doc_string = func.__doc__
                    if not doc_string is None and 'Measurement:' in doc_string:
                        desc = re.search('Description\:([\s\S]+?)Input', doc_string).groups()[0].strip()
                        print('{}) {}: {}'.format(count + 1, f, desc))
                        count += 1


    def select_data(self,nodes=[], communities=[]):
        """
        Subset the data based on the given communities or pieces of content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: New DataFrame with the select communities/content only
        """

        if len(nodes) > 0:
            if nodes=='all':
                data = self.dataset.copy()
            else:
                data = self.dataset.loc[self.dataset[self.content_col].isin(nodes)]
        elif len(communities) > 0:
            data = self.community_set.loc[self.community_set[self.community_col].isin(communities)]
        else:
            data = self.dataset.copy()
        return data

    def get_time_diff_granularity(self,time_diff,granularity):

        return time_diff / np.timedelta64(1,granularity)

    def get_time_granularity(self,time,time_unit):

        if time_unit.lower() == 'm':
            time /= 60.0
        if time_unit.lower() == 'h':
            time /= 60.0*60.0
        if time_unit.lower() == 'd':
            time /= 60.0*60.0*24.0

        return(time)

    def preprocess(self, node_level, nodes, community_level, communities):

        if node_level and len(nodes) == 0:
           nodes = self.node_list
        elif node_level and nodes == "all":
           nodes = self.dataset[self.content_col].unique()
        elif not node_level:
           nodes = []

        if community_level and len(communities) == 0:
           communities = self.community_list
        elif community_level and communities == "all":
           communities = self.community_set[self.community_col].dropna().unique()
        elif not community_level:
           communities = []

        data = self.select_data(nodes, communities)

        return(data)

    def order_of_spread(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Measurement: order_of_spread (population_order_of_spread, community_order_of_spread, node_order_of_spread)

        Description: Determine the order of spread between platforms of content

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: Dataframe of normalized counts of each platform in each position
                If community level : a dictionary mapping between each community
                                     and a dataframe of normalized counts
                If node level: a dictionary mapping the information to list of platforms in order
        """


        data = self.preprocess(node_level, nodes, community_level, communities)
        
        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)

        data = data.sort_values(self.timestamp_col)
        
        group_col = [self.content_col]
        if community_level:
            group_col += [self.community_col]

        data = data.drop_duplicates(subset=[self.content_col,self.platform_col])

        #create column with list of platforms in order of appearance of the info ID
        data = data.groupby(group_col)[self.platform_col].apply(list).reset_index()

        if not node_level:

            data = data[data[self.platform_col].apply(len) > 1]
            
            #for a group of information IDs, count what proportion of the time each platform is first, second, third, etc.
            def get_order_counts(grp):

                dfs = []
                for platform in platforms:
                
                    #get index of the platform in the ordered list (0  = 1st, 1 = 2nd, 2 = 3rd, -1 = not present)
                    grp[platform] = grp[self.platform_col].apply(lambda x: x.index(platform) if platform in x else -1)
                
                    #count the number of times in each position
                    value_counts = grp[grp[platform] != -1][platform].value_counts().reset_index()
                    value_counts['platform'] = platform
                    value_counts = value_counts.rename(columns={'index':'rank',
                                                                platform:'count'})

                    dfs.append(value_counts)

                grp = pd.concat(dfs)

                #normalize by the total count across all platforms
                grp = pd.pivot_table(grp,index='platform',values='count',columns='rank').fillna(0)
                grp = grp.div(grp.sum(axis=1),axis=0)
                cols = grp.columns
                
                if len(grp) > 0:
                    grp = pd.melt(grp.reset_index(),id_vars=['platform'],value_vars=cols)

                return(grp)

            if not community_level:
                data = get_order_counts(data)
            else:
                data = data.groupby(self.community_col).apply(get_order_counts).reset_index()
                
                data = dict(tuple(data.groupby(self.community_col)))
                data = {k:v[['platform','rank','value']] for k,v in data.items()}
        else:
            data = data.set_index(self.content_col)[self.platform_col]
            data = data.to_dict()

        if len(data) > 0:
            return(data)
        else:
            return(None)


    def time_delta(self, time_granularity="D", node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Measurement: time_delta (population_time_delta, community_time_delta, node_time_delta)

        Description: Determine the amount of time (default=days) it takes for a piece of information to appear on another platform.

        Input:
            time_granularity: Unit of time in which to measure
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the time difference from one platform to another
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to the time differences between platforms
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)

        if not community_level:
            group_col = [self.content_col, self.platform_col]
        else:
            group_col = [self.content_col, self.community_col, self.platform_col]


        data.sort_values(by=[self.timestamp_col], inplace=True, ascending=True)

        data.drop_duplicates(subset=group_col, inplace=True)
                
        group_col = [c for c in group_col if c != self.platform_col]
        

        #get all pairs of timestamps in each group
        data_combinations = data.groupby(group_col)[self.timestamp_col].apply(combinations,2).apply(list)
        lengths = data_combinations.apply(len)
        if (lengths > 0).sum() == 0:
            return None

        data_combinations = data_combinations.apply(pd.Series).stack().apply(pd.Series)
        data_combinations.columns = [self.timestamp_col,'next_platform_timestamp']
        #get time differences for each pair of time stamps
        data_combinations['time_diff'] = data_combinations['next_platform_timestamp'] - data_combinations[self.timestamp_col]
        data_combinations = data_combinations.reset_index()

        #merge to get first platform
        data_combinations = data_combinations.merge(data[group_col + [self.platform_col,self.timestamp_col]],
                                                    on=group_col + [self.timestamp_col],how='left')

        #merge to get second platform
        data = data_combinations.merge(data[group_col + [self.platform_col,self.timestamp_col]],
                                       right_on=group_col + [self.timestamp_col],
                                       left_on=group_col + ['next_platform_timestamp'],
                                       how='left',suffixes=('_first','_second'))

        data = data[group_col + ['platform_first','platform_second','time_diff']]
        data = data.rename(columns={'time_diff':'value'})

        data['value'] = data['value'].apply(lambda x: self.get_time_diff_granularity(x,time_granularity))

        if node_level:
            data = dict(tuple(data.groupby(self.content_col)))
        elif community_level:
            data = dict(tuple(data[[self.community_col,'platform_first','platform_second','value']].groupby(self.community_col)))
        else:
            data = data[['platform_first','platform_second','value']]
            
        return(data)


    def overlapping_users(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Measurement: overlapping_users (population_overlapping_users, community_overlapping_users, node_overlapping_users)        

        Description: Calculate the percentage of users common to all platforms (that share in a community/content)

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing each pair of platforms and the percentage of common users
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to a dataframe
        """
 
        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       
 
        data.loc[:,'values'] = 1
        data = data.drop_duplicates(subset=[self.user_col, self.platform_col])
        cols = [self.user_col, self.platform_col, 'values']
        index_cols = [self.user_col]


        if community_level:
            cols = [self.user_col, self.platform_col, self.community_col, 'values']
            index_cols = [self.user_col, self.community_col]
            group_col = self.community_col
        elif node_level:
            cols = [self.user_col, self.platform_col, self.content_col, 'values']
            index_cols = [self.user_col, self.content_col]
            group_col = self.content_col
        else:
            group_col = []


        user_platform = data[cols].pivot_table(index=index_cols,
                                               columns=self.platform_col,
                                               values='values').fillna(0)
        user_platform = user_platform.astype(bool)
        
        platforms = list(user_platform.columns.values)

        def get_meas(grp):
            pl_1, pl_2, val = [], [], []
            for i, p1 in enumerate(platforms):
                for j, p2 in enumerate(platforms):
                    if p1 != p2:
                        x = (grp[p1] & grp[p2]).sum()
                        total = float(grp[p1].sum())
                        if total > 0:
                            x = x / total
                        else:
                            x = 0
                        pl_1.append(p1)
                        pl_2.append(p2)
                        val.append(x)

            meas = pd.DataFrame({"platform_1": pl_1, "platform_2": pl_2, "value": val})
            meas = meas.drop_duplicates(subset=["platform_1", "platform_2"])
            return meas

        if node_level or community_level:
            meas = {}
            for i, grp in user_platform.groupby(group_col):
                meas[i] = get_meas(grp)
        else:
            meas = get_meas(user_platform)
        return meas

    def size_of_audience(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Measurement: size_of_audience (population_size_of_audience, community_size_of_audience, node_size_of_audience)
        
        Description: The ranking of audience sizes on each platform

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing each platform and audience size
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to a dataframe
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)

        group_col = [self.content_col]
        if community_level:
            group_col = [self.content_col,self.community_col]

        aud = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col)[self.user_col].nunique()).reset_index()

        if 'nodeUserID' not in aud.columns:
            aud = pd.melt(aud,id_vars=self.platform_col)

        aud.columns = [self.platform_col] + group_col + ['value']
        aud = aud[aud['value'] != -1]
        
        if node_level:
            counts = aud.groupby(self.content_col)[self.platform_col].count()
            counts.name = 'count'
            counts = counts.reset_index()
            aud = aud.merge(counts,on=self.content_col)
            aud = aud[aud['count'] > 1]

            aud = dict(tuple(aud.groupby(self.content_col)))
            aud = {k:v[[self.platform_col,'value']] for k,v in aud.items()}
        elif community_level:
            aud = aud.groupby([self.platform_col,self.community_col])['value'].mean().reset_index()
            aud = dict(tuple(aud.groupby(self.community_col)))
            aud = {k:v[[self.platform_col,'value']] for k,v in aud.items()}
        else:
            aud = aud[[self.platform_col,'value']].groupby(self.platform_col).mean().reset_index()

        return aud


    def speed_of_spread(self, time_unit='h', node_level = False, community_level = False, 
                        nodes=[], communities=[],):
        """
        Measurement: speed_of_spread (population_speed_of_spread, community_speed_of_spread, node_speed_of_spread)
        Description: Determine the speed at which the information is spreading

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the platform and average speed
                If community level, a dictionary mapping each community to dataframe
                If node level, a dictionary mapping each node to a dataframe
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       
        group_col = [self.content_col]
        if community_level:
            group_col = [self.content_col,self.community_col]

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).total_seconds()
            time = self.get_time_granularity(time,time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        speeds = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col).apply(get_speed)).reset_index()
        
        if 0 not in speeds.columns:
            speeds = pd.melt(speeds,id_vars=self.platform_col)

        speeds.columns = [self.platform_col] + group_col + ['value']
        speeds = speeds[speeds['value'] != -1]
        
        if node_level:
            counts = speeds.groupby(self.content_col)[self.platform_col].count()
            counts.name = 'count'
            counts = counts.reset_index()
            speeds = speeds.merge(counts,on=self.content_col)
            speeds = speeds[speeds['count'] > 1]

            speeds = dict(tuple(speeds.groupby(self.content_col)))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        elif community_level:
            speeds = speeds.groupby([self.platform_col,self.community_col])['value'].mean().reset_index()
            speeds = dict(tuple(speeds.groupby(self.community_col)))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        else:
            speeds = speeds[[self.platform_col,'value']].groupby(self.platform_col).mean().reset_index()

        return speeds

    def size_of_shares(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Measurement: size_of_shares (population_size_of_shares, community_size_of_shares, community_size_of_shares, node_size_of_shares)

        Description: Determine the number of shares on each platform

        Input:
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the platform and average number of shares
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to a dataframe
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       
        group_col = [self.content_col]
        if community_level:
            group_col += [self.community_col]

        share_counts = data.groupby([self.platform_col] + group_col)[self.timestamp_col].count().reset_index()
        share_counts.columns = [self.platform_col] + group_col + ['value']
        share_counts = share_counts.sort_values('value',ascending=False)

        if node_level:
            counts = share_counts.groupby(self.content_col)[self.platform_col].count()
            counts.name = 'count'
            counts = counts.reset_index()
            share_counts = share_counts.merge(counts,on=self.content_col)
            share_counts = share_counts[share_counts['count'] > 1]


            share_counts = dict(tuple(share_counts.groupby(self.content_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        elif community_level:
            share_counts = dict(tuple(share_counts.groupby([self.platform_col,self.community_col])['value'].mean().reset_index().groupby(self.community_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        else:
            share_counts = share_counts.groupby(self.platform_col)['value'].mean().reset_index()

        return share_counts

    def temporal_correlation(self, measure="share", time_granularity="D",node_level = False, community_level = False, 
                             nodes=[], communities=[]):
        """
        Measurement: temporal_correlation_share (population_temporal_correlation_share, community_correlation_share, node_correlation_share), temporal_correlation_audience (population_temporal_correlation_audience, community_correlation_audience, node_correlation_audience)

        Description: Calculates the correlation between the activity time series  between all pairs of platforms

        Input:
            measure: What to measure, number of shares ("share") or audience growth ("audience")
            time_graularity: the scale on which to aggregate activity
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the pearson correlation value between any two platforms
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to a dataframe
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       
        data = data.set_index(self.timestamp_col)
        
        group_col = [self.content_col]
        if community_level:
            group_col += [self.community_col]

        if measure == 'share':
            data = data.groupby(group_col + [self.platform_col,pd.Grouper(freq=time_granularity)])['nodeUserID'].count().reset_index()
        else:
            data = data.groupby(group_col + [self.platform_col,pd.Grouper(freq=time_granularity)])['nodeUserID'].nunique().reset_index()

        data = data.rename(columns={'nodeUserID':'value'})

        data = pd.pivot_table(data,values='value',index= group_col + [self.timestamp_col],columns=self.platform_col).fillna(0).reset_index()
                
        dfs = []
        for i, p1 in enumerate(platforms):
            for j,p2 in enumerate(platforms[i+1:]):

                counts = data.groupby(group_col)[self.timestamp_col].count().reset_index()
                counts.columns = group_col + ['count']

                df = data.merge(counts,on=group_col)
                
                df = df[df['count'] > 1]

                df = df.groupby(group_col).apply(lambda x: pearsonr(x[p1],x[p2])[0]).reset_index()
                df.columns = group_col + ['value']
                df['platform1'] = p1
                df['platform2'] = p2
                
                dfs.append(df)
                
        data = pd.concat(dfs)
        data = data.dropna()

        if node_level:
            corr = dict(tuple(data.groupby(self.content_col)))
        elif community_level:
            data = data.drop(self.content_col,axis=1)
            corr = dict(tuple(data.groupby([self.community_col])))
            corr = {k:v.drop(self.community_col,axis=1) for k,v in corr.items()}
        else:
            corr = data.drop(self.content_col,axis=1)

        return corr

        

    def lifetime_of_spread(self, node_level = False, community_level = False, 
                        nodes=[], communities=[],time_unit='D'):

        """
        Measurement: lifetime_of_spread (population_lifetime_of_spread, community_lifetime_of_spread, node_lifetime_of_spread)

        Description: Ranks the different platforms based on the lifespan (default = days) of the content/community

        Input:
            time_unit: the unit of time to measure lifetime
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the average lifetime for each platform
                If community level, a dictionary mapping each community to a dataframe
                If node level, a dictionary mapping each node to a dataframe
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       
        group_col = [self.content_col]
        if community_level:
            group_col = [self.content_col,self.community_col]

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).total_seconds()
            time = self.get_time_granularity(time,time_unit)
            return time

        lifetimes = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col).apply(get_lifetime)).reset_index()
        
        if 0 not in lifetimes.columns:
            lifetimes = pd.melt(lifetimes,id_vars=self.platform_col)

        lifetimes.columns = [self.platform_col] + group_col + ['value']
        lifetimes = lifetimes[lifetimes['value'] != -1]
        
        if node_level:
            counts = lifetimes.groupby(self.content_col)[self.platform_col].count()
            counts.name = 'count'
            counts = counts.reset_index()
            lifetimes = lifetimes.merge(counts,on=self.content_col)
            lifetimes = lifetimes[lifetimes['count'] > 1]

            lifetimes = dict(tuple(lifetimes.groupby(self.content_col)))
            lifetimes = {k:v[[self.platform_col,'value']] for k,v in lifetimes.items()}
        elif community_level:
            lifetimes = lifetimes.groupby([self.platform_col,self.community_col])['value'].mean().reset_index()

            lifetimes = dict(tuple(lifetimes.groupby(self.community_col)))
            lifetimes = {k:v[[self.platform_col,'value']] for k,v in lifetimes.items()}
        else:
            lifetimes = lifetimes[[self.platform_col,'value']].groupby(self.platform_col).mean().reset_index()

        return lifetimes


    def correlation_of_information(self, measure="share", time_unit='D',
                                   community_level=False,communities=[]):
        """
        Measurement: correlation_of_audiences, correlation_of_lifetimes, correlation_of_shares, correlation_of_speeds

        Description: Compute Pearson correlation
            1. Correlation between shares of information across platforms
            2. Correlation between audience sizes
            3. Correlation between lifetimes (default = days) of information across platforms
            4. Correlation between speeds (default = days) of information across platforms

        Input:
            measure: What to measure: number of shares, audience, lifetime, or speed?
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level, a dataframe containing the correlation between platforms for each node
                If community level, a dictionary mapping each community to a dataframe
                If node level, same as population level
        """

        if community_level and len(communities) == 0:
           communities = self.community_list
        elif community_level and communities == "all":
           communities = self.community_set[self.community_col].unique()
        elif not community_level:
           communities = []

        data = self.select_data(communities=communities)

        platforms = sorted(data[self.platform_col].unique())

        if len(platforms) <= 1:
            warnings.warn("Not enough platforms for cross-platform measurements")
            return(None)
       

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).total_seconds()
            time = self.get_time_granularity(time,time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).total_seconds()
            time = self.get_time_granularity(time,time_unit)
            return time

        platforms = sorted(data[self.platform_col].unique())

        group_col = [self.content_col, self.platform_col]
        if community_level:
            group_col += [self.community_col]

        if measure == 'share':
            data = data.groupby(group_col)['nodeUserID'].count()
        elif measure == 'audience':
            data = data.groupby(group_col)['nodeUserID'].nunique()
        elif measure == 'lifetime':
            data = data.groupby(group_col).apply(get_lifetime)
        elif measure == 'speed':
            data = data.groupby(group_col).apply(get_speed)

        data.name = 'value'
        data = data.reset_index()


        def get_ranking_correlations(grp):

            grp = pd.pivot_table(grp,index=self.platform_col,values='value',columns=self.content_col).fillna(0)

            platform1s = []
            platform2s = []
            corrs = []

            for i,p1 in enumerate(platforms):
                for p2 in platforms[i+1:]:
                
                    if p1 in grp.index and p2 in grp.index:
                        corr = spearmanr(grp.loc[p1].values,grp.loc[p2].values)[0]

                        platform1s.append(p1)
                        platform2s.append(p2)
                        corrs.append(corr)
            
            corr = pd.DataFrame({'platform1':platform1s,'platform2':platform2s,'value':corrs})
            
            return corr

        if not community_level:
            corr = get_ranking_correlations(data)
        else:
            
            corr = data.groupby(self.community_col).apply(get_ranking_correlations).reset_index()
            corr = dict(tuple(corr.groupby(self.community_col)))
            corr = {k:v[['platform1','platform2','value']] for k,v in corr.items()}

        return corr
