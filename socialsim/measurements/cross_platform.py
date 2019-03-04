import sys
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from itertools import combinations
import pprint

from .measurements import MeasurementsBaseClass


class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata=None,
                 platform_col="platform", timestamp_col="nodeTime",
                 user_col="nodeUserID", content_col="informationID",
                 community_col="community", log_file='cross_platform_measurements_log.txt',
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
        self.dataset = dataset
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform_col = platform_col
        self.content_col = content_col
        self.community_col = community_col


        print(self.dataset)

        self.measurement_type = 'cross_platform'

        if metadata is None:
            self.community_set = self.dataset
            self.community_set[self.community_col] = "Default Community"
            self.community_set[self.community_col] = np.random.choice(['A','B','C'],len(self.dataset))

        else:
            self.community_set = metadata.communities

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

    def select_data(self, nodes=None, communities=None):
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

    def time_granularity(self,time_diff,granularity):

        if granularity.lower() == 's':
            return time_diff.seconds
        elif granularity.lower() == 'm':
            return time_diff.minutes
        elif granularity.lower() == 'h':
            return time_diff.hours
        elif granularity.lower() == 'd':
            return time_diff.days

    def order_of_spread(self, nodes=None, communities=None):
        """
        Determine the order of spread between platforms of a community/content
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a dictionary mapping each platform to an array of percent freq. in each rank
                If community, a nested dictionary mapping each community to a dictionary mapping each platform
                    to an array of percent freq. in each rank
                Else, a dictionary mapping between the content to the ranked list of platforms
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].unique()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()

        data = self.select_data(nodes, communities)
        platforms = sorted(data[self.platform_col].unique())

        def platform_order(diction):
            plat_diction = {p: np.zeros((len(platforms))) for p in platforms}
            for _, plat in diction.items():
                for p in platforms:
                    pos, = np.where(plat == p)
                    plat_diction[p][pos] += 1
            for k, v in plat_diction.items():
                if v.sum(axis=0) == 0:
                    plat_diction[k] = [0] * len(platforms)
                else:
                    plat_diction[k] = v / v.sum(axis=0)
            return plat_diction

        if len(communities) > 0:
            data.drop_duplicates(subset=[self.community_col, self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.community_col).apply(
                lambda x: x.groupby(self.content_col).apply(lambda y: y[self.platform_col].values).to_dict())
            community_platform_order = data.to_dict()
            keywords_to_order = {}
            for comm, content_diction in community_platform_order.items():
                keywords_to_order[comm] = platform_order(content_diction)
            plt_1, position, val = [], [], []
            for k, v in keywords_to_order.items():
                plt_1.extend([k] * len(v))
                position.extend([1, 2, 3])
                val.extend(v)
            return pd.DataFrame({"platform": plt_1, "position": position, "value": val})
        else:
            data.drop_duplicates(subset=[self.content_col, self.platform_col], inplace=True)
            data = data.groupby(self.content_col).apply(lambda x: x[self.platform_col].tolist())
            n_platforms = data.apply(len)
            data = data[n_platforms > 1] 
            keywords_to_order = data.to_dict()

            if len(nodes) == 0 and len(communities) == 0:
                plt_1, position, val = [], [], []
                keywords_to_order = platform_order(keywords_to_order)
                for k, v in keywords_to_order.items():
                    plt_1.extend([k] * len(v))
                    position.extend([1, 2, 3])
                    val.extend(v)
                return pd.DataFrame({"platform": plt_1, "position": position, "value": val})
            return keywords_to_order

    def time_delta(self, time_granularity="s", nodes=None, communities=None):
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

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()


        data = self.select_data(nodes, communities)


        if len(communities) == 0:
            group_col = [self.content_col, self.platform_col]
        else:
            group_col = [self.content_col, self.community_col, self.platform_col]

        data.drop_duplicates(subset=group_col, inplace=True)
        data.sort_values(by=[self.timestamp_col], inplace=True, ascending=True)
                
        print(data[group_col + [self.timestamp_col]])

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

        data['value'] = data['value'].apply(lambda x: self.time_granularity(x,time_granularity))

        if len(nodes) > 0:
            data = dict(tuple(data.groupby(self.content_col)))
        elif len(communities) > 0:
            data = dict(tuple(data[[self.community_col,'platform_first','platform_second','value']].groupby(self.community_col)))
        else:
            data = data[['platform_first','platform_second','value']]
            
        return(data)


    def overlapping_users(self, nodes=None, communities=None):
        """
        Calculate the percentage of users common to all platforms (that share in a community/content)
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a matrix of percentages of common users to any pair of platforms.
                Else, a dictionary mapping a community/content to a matrix of the percentages of common users that
                share in that community/content across all pairs of platforms
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()
        data = self.select_data(nodes, communities)
 
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
        
        print(user_platform)

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

    def size_of_audience(self, nodes=None, communities=None):
        """
        Determine the ranking of audience sizes on each platform
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a ranked list of the platforms with the largest audience sizes
                Else, a dictionary mapping the community/content to a ranked list of the platforms with the largest
                    audience sizes.
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()
        data = self.select_data(nodes, communities)

        if len(communities) > 0:
            group_col = self.community_col
        elif len(nodes) > 0:
            group_col = self.content_col
        else:
            group_col = self.platform_col

        def audience(grp):
            aud = grp.groupby(self.platform_col).apply(lambda x: len(x[self.user_col].unique())).to_dict()
            return [[item[0], item[1]] for item in sorted(aud.items(), reverse=True, key=lambda kv: kv[1])]

        if len(nodes) == 0 and len(communities) == 0:
            return audience(data)
        else:
            audience_diction = data.groupby(group_col).apply(audience).to_dict()
            final_diction = {}
            for name, sorted_list in audience_diction.items():
                platform_list, value = [], []
                for i in sorted_list:
                    platform_list.append(i[0])
                    value.append(i[1])
                final_diction[name] = pd.DataFrame({"platform": platform_list, "value": value})
            return final_diction

    def speed_of_spread(self, nodes=None, communities=None):

        """
        Determine the speed at which the information is spreading
        :param nodes: List of nodes
        :param communities: List of communities
        :return: If population, a DataFrame with columns platform and value
                If community, a dictionary mapping a community to a a DataFrame with columns platform and value
                Else, a dictionary mapping each content to the ranked list of platforms on which it spreads the fastest
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()
        data = self.select_data(nodes, communities)

        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds / 60.0 / 60.0
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

    def size_of_shares(self, nodes=None, communities=None):
        """
        Determine the number of shares per platform
        :param nodes: List of specific content
        :param communities: List of communities
        :return: If population, a ranked list of platforms based on total activity
                Else, a dictionary mapping the community/content to a ranked list of platforms based on activity
        """

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()
        data = self.select_data(nodes, communities)


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

    def temporal_correlation(self, measure="share", time_granularity="D",
                             nodes=None, communities=None):
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

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()
        
        data = self.select_data(nodes, communities)

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

        

    def lifetime_of_spread(self, nodes=None, communities=None,time_unit='H'):
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

        if nodes is None:
            nodes = self.node_list
        elif nodes == "all":
            nodes = self.dataset[self.content_col].tolist()
        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()

        data = self.select_data(nodes, communities)


        group_col = [self.content_col]
        if len(communities) > 0:
            group_col = [self.content_col,self.community_col]

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            if time_unit == 'M':
                time /= 60.0
            if time_unit == 'H':
                time /= 60.0*60.0
            if time_unit == 'D':
                time /= 60.0*60.0*24.0

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


    def correlation_of_information(self, measure="share", communities=None,time_unit='H'):
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

        if communities is None:
            communities = self.community_list
        elif communities == "all":
            communities = self.community_set[self.community_col].unique()

        data = self.select_data(communities=communities)


        def get_speed(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = get_granularity(time,time_unit)
            if time == 0:
                speed = -1
            else:
                speed = len(grp) / time
            return speed

        def get_lifetime(grp):
            time = (grp[self.timestamp_col].max() - grp[self.timestamp_col].min()).seconds
            time = get_granularity(time,time_unit)
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
