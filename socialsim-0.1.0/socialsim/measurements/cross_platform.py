import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from itertools import combinations
import pprint

from .measurements import MeasurementsBaseClass

from ..utils import add_communities_to_dataset


class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata=None, 
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
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)

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
        else:
            community_directory = metadata.community_directory
            self.dataset = add_communities_to_dataset(dataset, 
                community_directory)

            self.community_set = self.dataset['community'].unique()

        if node_list == "all":
            self.node_list = self.dataset[self.content_col].tolist()
        elif node_list is not None:
            self.node_list = node_list
        else:
            self.node_list = []

        if self.community_set is not None:
            if community_list == "all":
                self.community_list = self.community_set[self.community_col].unique()
            elif community_list is not None:
                self.community_list = community_list
            else:
                self.community_list = []
        else:
            self.community_list = []

        return None


    def select_data(self, node_level = False, community_level = False, 
        nodes=[], communities=[]):
        """
        Subset the data based on the given communities or pieces of content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: New DataFrame with the select communities/content only
        """

        if nodes is None:
            nodes = self.node_list
        if communities is None:
            communities = self.community_list

        if len(nodes) > 0:
            data = self.dataset.loc[self.dataset[self.content_col].isin(nodes)]
        elif len(communities) > 0:
            data = self.community_set.loc[self.community_set[self.community_col].isin(communities)]
        else:
            data = self.dataset.copy()
        return data

    def get_time_diff_granularity(self,time_diff,granularity):

        if granularity.lower() == 's':
            return time_diff.seconds
        elif granularity.lower() == 'm':
            return time_diff.minutes
        elif granularity.lower() == 'h':
            return time_diff.hours
        elif granularity.lower() == 'd':
            return time_diff.days

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
        if community_level and len(communities) == 0:
            communities = self.community_list
        elif community_level and communities == "all":
            communities = self.community_set[self.community_col].unique()

        data = self.select_data(nodes, communities)

        return(data)

    def order_of_spread(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a dictionary mapping each platform to an array of percent freq. in each rank
                If community, a nested dictionary mapping each community to a dictionary mapping each platform
                    to an array of percent freq. in each rank
                Else, a dictionary mapping between the content to the ranked list of platforms
        """

        data = self.preprocess(node_level, nodes, community_level, communities)
        
        platforms = sorted(data[self.platform_col].unique())

        data = data.sort_values(self.timestamp_col)
        
        group_col = [self.content_col]
        if len(communities) > 0:
            group_col += [self.community_col]

        data = data.drop_duplicates(subset=[self.content_col,self.platform_col])

        #create column with list of platforms in order of appearance of the info ID
        data = data.groupby(group_col)[self.platform_col].apply(list).reset_index()

        if len(nodes) == 0:

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
                grp = grp / grp.sum(axis=0)
                cols = grp.columns
                
                grp = pd.melt(grp.reset_index(),id_vars=['platform'],value_vars=cols)

                return(grp)

            if len(communities) == 0:
                data = get_order_counts(data)
            else:
                data = data.groupby(self.community_col).apply(get_order_counts).reset_index()
                
                data = dict(tuple(data.groupby(self.community_col)))
                data = {k:v[['platform','rank','value']] for k,v in data.items()}
        else:
            data = data.set_index(self.content_col)[self.platform_col]
            data = data.to_dict()

        return(data)


    def time_delta(self, time_granularity="s", node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Determine the amount of time it takes for a community/content to appear on another platform
        :param time_granularity: Unit of time to calculate {s=seconds, m=minutes, h=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a DataFrame with columns platform_1, platform_2, and value
                If community, a dictionary mapping a community to a DataFrame with columns
                    platform_1, platform_2, and value
                Else, a dictionary mapping a content to a list (equal in length to the number of platforms)
                of the time passed since the first time that content was observed. This list is sorted alphabetically
                to preserve the same ordering of platforms in all cases. This can causes negative times.
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        if len(communities) == 0:
            group_col = [self.content_col, self.platform_col]
        else:
            group_col = [self.content_col, self.community_col, self.platform_col]

        data.drop_duplicates(subset=group_col, inplace=True)
        data.sort_values(by=[self.timestamp_col], inplace=True, ascending=True)
                
        group_col = [c for c in group_col if c != self.platform_col]

        #get all pairs of timestamps in each group
        data_combinations = data.groupby(group_col)[self.timestamp_col].apply(combinations,2).apply(list).apply(pd.Series).stack().apply(pd.Series)
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

        if len(nodes) > 0:
            data = dict(tuple(data.groupby(self.content_col)))
        elif len(communities) > 0:
            data = dict(tuple(data[[self.community_col,'platform_first','platform_second','value']].groupby(self.community_col)))
        else:
            data = data[['platform_first','platform_second','value']]
            
        return(data)


    def overlapping_users(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Calculate the percentage of users common to all platforms (that share in a community/content)
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of percentages of common users to any pair of platforms.
                Else, a dictionary mapping a community/content to a matrix of the percentages of common users that
                share in that community/content across all pairs of platforms
        """
 
        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())
        
        data['values'] = 1
        data = data.drop_duplicates(subset=[self.user_col, self.platform_col])
        cols = [self.user_col, self.platform_col, 'values']
        index_cols = [self.user_col]

        if len(communities) > 0:
            cols = [self.user_col, self.platform_col, self.community_col, 'values']
            index_cols = [self.user_col, self.community_col]
            group_col = self.community_col
        elif len(nodes) > 0:
            cols = [self.user_col, self.platform_col, self.content_col, 'values']
            index_cols = [self.user_col, self.content_col]
            group_col = self.content_col
        else:
            group_col = []

        user_platform = data[cols].pivot_table(index=index_cols,
                                               columns=self.platform_col,
                                               values='values').fillna(0)
        user_platform = user_platform.astype(bool)
        
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

        if len(nodes) != 0 or len(communities) != 0:
            meas = {}
            for i, grp in user_platform.groupby(group_col):
                meas[i] = get_meas(grp)
        else:
            meas = get_meas(user_platform)
        return meas

    def size_of_audience(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Determine the ranking of audience sizes on each platform
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a ranked list of the platforms with the largest audience sizes
                Else, a dictionary mapping the community/content to a ranked list of the platforms with the largest
                    audience sizes.
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        aud = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col)[self.user_col].nunique()).reset_index()

        if 'nodeUserID' not in aud.columns:
            aud = pd.melt(aud,id_vars=self.platform_col)

        aud.columns = [self.platform_col] + group_col + ['value']
        aud = aud[aud['value'] != -1]
        
        if len(nodes) > 0:
            aud = dict(tuple(aud.groupby(self.content_col)))
            aud = {k:v[[self.platform_col,'value']] for k,v in aud.items()}
        elif len(communities) > 0:
            aud = dict(tuple(aud.groupby(self.community_col)[[self.platform_col,'value']]))
            aud = {k:v[[self.platform_col,'value']] for k,v in aud.items()}
        else:
            aud = aud[[self.platform_col,'value']]

        return aud


    def speed_of_spread(self, time_unit='h', node_level = False, community_level = False, 
                        nodes=[], communities=[],):

        """
        Determine the speed at which the information is spreading
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a DataFrame with columns platform and value
                If community, a dictionary mapping a community to a a DataFrame with columns platform and value
                Else, a dictionary mapping each content to the ranked list of platforms on which it spreads the fastest
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
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
        
        if len(nodes) > 0:
            speeds = dict(tuple(speeds.groupby(self.content_col)))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        elif len(communities) > 0:
            speeds = dict(tuple(speeds.groupby(self.community_col)[[self.platform_col,'value']]))
            speeds = {k:v[[self.platform_col,'value']] for k,v in speeds.items()}
        else:
            speeds = speeds[[self.platform_col,'value']]

        return speeds

    def size_of_shares(self, node_level = False, community_level = False, 
                        nodes=[], communities=[]):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """


        data = self.preprocess(node_level, nodes, community_level, communities)


        group_col = [self.content_col]
        if len(communities) > 0:
            group_col += [self.community_col]

        share_counts = data.groupby([self.platform_col] + group_col)[self.timestamp_col].count().reset_index()
        share_counts.columns = [self.platform_col] + group_col + ['value']
        share_counts = share_counts.sort_values('value',ascending=False)

        if len(nodes) > 0:
            share_counts = dict(tuple(share_counts.groupby(self.content_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        elif len(communities) > 0:
            share_counts = dict(tuple(share_counts.groupby([self.platform_col,self.community_col])['value'].mean().reset_index().groupby(self.community_col)))
            share_counts = {k:v[[self.platform_col,'value']] for k,v in share_counts.items()}
        else:
            share_counts = share_counts.groupby(self.platform_col)['value'].mean().reset_index()

        return share_counts

    def temporal_correlation(self, measure="share", time_granularity="D",node_level = False, community_level = False, 
                             nodes=[], communities=[]):
        """
        Calculates the correlation between the activity over time between all pairs of platforms
                Github | Reddit | Twitter
        ---------------------------------
        Github | 1.0
        ---------------------------------
        Reddit |           1.0
        ---------------------------------
        Twitter|                   1.0
        ---------------------------------
        :param measure: What to measure: number of shares or audience growth?
        :param time_granularity: The scale on which to aggregate activity {S=seconds, M=minutes, H=hours, D=days}
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of pearson correlations between platforms.
                    Else, a dictionary mapping a community/content to the matrix of correlations
        """

        data = self.preprocess(node_level, nodes, community_level, communities)

        platforms = sorted(data[self.platform_col].unique())

        data = data.set_index(self.timestamp_col)
        
        group_col = [self.content_col]
        if len(communities) > 0:
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
                df = data.groupby(group_col).apply(lambda x: pearsonr(x[p1],x[p2])[0]).reset_index()
                df.columns = group_col + ['value']
                df['platform1'] = p1
                df['platform2'] = p2

                dfs.append(df)
                
        data = pd.concat(dfs)
        data = data.dropna()

        if len(nodes) > 0:
            corr = dict(tuple(data.groupby(self.content_col)))
        elif len(communities) > 0:
            data = data.drop(self.content_col,axis=1)
            corr = dict(tuple(data.groupby([self.community_col])))
            corr = {k:v.drop(self.community_col,axis=1) for k,v in corr.items()}
        else:
            corr = data.drop(self.content_col,axis=1)

        return corr

        

    def lifetime_of_spread(self, node_level = False, community_level = False, 
                        nodes=[], communities=[],time_unit='H'):
        """
        Ranks the different platforms based on the lifespan of content/community/population
        :param nodes: List of specific content
        :param communities: List of communities
        :param time_unit: Unit of time to measure lifetime, e.g. 'S','M','H','D'
        :return: If population, a DataFrame (columns = platform, value)
                If community, a  dictionary mapping each community to a DataFrame (columns= platform, value)
                If nodes, returns a dictionary mapping each piece of information to a ranked list of platforms
                    (by longest lifespan).
                        Ex: {info_1: [github, twitter, reddit],
                            info_2: [reddit, twitter], ... }
        """


        data = self.preprocess(node_level, nodes, community_level, communities)


        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = self.get_time_granularity(time,time_unit)
            return time

        lifetimes = data.groupby(self.platform_col).apply(lambda x: x.groupby(group_col).apply(get_lifetime)).reset_index()
        
        if 0 not in lifetimes.columns:
            lifetimes = pd.melt(lifetimes,id_vars=self.platform_col)

        lifetimes.columns = [self.platform_col] + group_col + ['value']
        lifetimes = lifetimes[lifetimes['value'] != -1]
        
        if len(nodes) > 0:
            lifetimes = dict(tuple(lifetimes.groupby(self.content_col)))
            lifetimes = {k:v[[self.platform_col,'value']] for k,v in lifetimes.items()}
        elif len(communities) > 0:
            lifetimes = dict(tuple(lifetimes.groupby(self.community_col)[[self.platform_col,'value']]))
            lifetimes = {k:v[[self.platform_col,'value']] for k,v in lifetimes.items()}
        else:
            lifetimes = lifetimes[[self.platform_col,'value']]

        return lifetimes


    def correlation_of_information(self, measure="share", time_unit='H',
                                   community_level=False,communities=[]):
        """
        Compute Pearson correlation
        1. Correlation between shares of information across platforms
        2. Correlation between audience sizes
        3. Correlation between lifetimes of information pieces across platforms
        4. Correlation between speeds of information across platforms
        :param measure: What to measure: number of share, audience, lifetime, or speed?
        :param communities: List of communities
        :param time_unit: time unit for speed and lifetime measures
        :return: If population, a matrix of correlations between all platforms based on the measure provided
                If community, a dictionary mapping each community to a matrix of correlations between all platforms
                    based on the measure provided.
        """

        if community_level and len(communities) == 0:
            communities = self.community_list
        elif community_level and communities == "all":
            communities = self.community_set[self.community_col].unique()

        data = self.select_data(communities=communities)

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = self.get_time_granularity(time,time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = self.get_time_granularity(time,time_unit)
            return time

        platforms = sorted(data[self.platform_col].unique())

        group_col = [self.content_col, self.platform_col]
        if len(communities) > 0:
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
                
                    corr = spearmanr(grp.loc[p1].values,grp.loc[p2].values)[0]

                    platform1s.append(p1)
                    platform2s.append(p2)
                    corrs.append(corr)
            
            corr = pd.DataFrame({'platform1':platform1s,'platform2':platform2s,'value':corrs})
            
            return corr

        if len(communities) == 0:
            corr = get_ranking_correlations(data)
        else:
            
            corr = data.groupby(self.community_col).apply(get_ranking_correlations).reset_index()
            corr = dict(tuple(corr.groupby(self.community_col)))
            corr = {k:v[['platform1','platform2','value']] for k,v in corr.items()}

        return corr
