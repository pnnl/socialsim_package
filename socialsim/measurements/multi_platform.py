import pandas as pd
import numpy as np
import pprint
import warnings
import re

from collections import Counter, OrderedDict
from .measurements import MeasurementsBaseClass
from ..utils import add_communities_to_dataset
import re

class MultiPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration={}, metadata=None, platform_col="platform",
                 timestamp_col="nodeTime", user_col="nodeUserID", content_col="informationID",
                 community_col="community", node_list=None, community_list=None, id_col='nodeID',
                 root_col="rootID", event_col="actionType", log_file='multi_platform_measurements_log.txt'):
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
        super(MultiPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)

        self.dataset = dataset
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform_col = platform_col
        self.content_col = content_col
        self.community_col = community_col
        self.id_col = id_col
        self.root_col = root_col
        self.event_col = event_col

        self.measurement_type = 'multi_platform'

        if metadata is None or (metadata.community_directory is None and metadata.communities is None):
            self.community_set = self.dataset
            self.community_set[self.community_col] = "Default Community"
        else:
            community_directory = metadata.community_directory
            communities = metadata.communities
            self.community_dict = communities
            self.community_set = add_communities_to_dataset(dataset,
                                                            community_directory,
                                                            communities)
            

        if metadata is None or metadata.node_list is None or metadata.node_list == 'all':
            if node_list == "all":
                self.node_list = self.dataset[self.content_col].tolist()
            elif node_list is not None:
                self.node_list = node_list
            else:
                self.node_list = []
        else:
            self.node_list = metadata.node_list

        if self.community_set is not None:
            if community_list is not None and len(community_list) > 0:
                self.community_list = community_list
            else:
                self.community_list = self.community_set[self.community_col].dropna().unique()
        else:
            self.community_list = []

        if metadata is None or metadata.previous_user_data is None:       
            self.previous_user_data = pd.DataFrame(columns=[self.user_col,self.content_col])
        else:
            self.previous_user_data = metadata.previous_user_data
            
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

    def select_data(self, nodes=[], communities=[], platform="all", action_types=[],date_range=[]):
        """
        Subset the data based on the given communities or pieces of content
        :param nodes: List of specific content
        :param communities: List of communities
        :param platform: Name of platform to include or "all" for all platforms
        :return: New DataFrame with the select communities/content only
        """
        
        if nodes is None:
            nodes = []
        if communities is None:
            communities = []
        if platform == "all":
            platform_list = self.dataset[self.platform_col].unique()
        else:
            platform_list = [platform]

        if len(nodes) > 0:
            if str(nodes) == 'all':
                data = self.dataset.loc[self.dataset[self.platform_col].isin(platform_list)]
            else:
                data = self.dataset.loc[(self.dataset[self.content_col].isin(nodes)) &
                                        (self.dataset[self.platform_col].isin(platform_list))]
        elif len(communities) > 0:
            data = self.community_set[(self.community_set[self.community_col].isin(communities)) &
                                      (self.community_set[self.platform_col].isin(platform_list))]
        else:
            data = self.dataset.loc[self.dataset[self.platform_col].isin(platform_list)]

        if len(action_types) > 0:
            data = data[data[self.event_col].isin(action_types)]            

        if len(date_range) == 2:
            data = data[ (data[self.timestamp_col] >= date_range[0] ) & (data[self.timestamp_col] < date_range[1]) ]

        return data


    def preprocess(self, node_level, nodes, community_level, communities, platform, action_types,date_range=[]):
        """
        Determine the node list and community list and ubset the data based on the given 
        communities or pieces of content
        :param node_level: Boolean indicating whether to subset based on nodes
        :param nodes: List of specific content
        :param community_level: Boolean indicating whether to subset based on communities
        :param communities: List of communities
        :param platform: Name of platform to include or "all" if all platforms
        :return: New DataFrame with the select communities/content/platform only
        """

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
        
    
        data = self.select_data(nodes, communities, platform=platform, action_types=action_types, date_range=date_range)

        return data

    def get_shares(self, x):
        """
        Count the number of events (shares)
        :param x: A Series with one element per share event
        :return: A scalar with the length of the series
        """

        return(x.count())

    def get_audience(self, x):
        """
        Count the number of users
        :param x: A Series of user IDs with one element per share event
        :return: A scalar with the number of unique elements in the Series
        """

        return(x.nunique())

    def get_lifetime(self,x, time_unit):
        """
        Calculate the time elapsed between first and last events
        :param x: A Series of datetime values with one element per share event
        :param time_unit: Granularity of time (e.g. "D","m","h")
        :return: A scalar with time elapsed between first and last events
        """

        time = (x.max() - x.min()) / np.timedelta64(1,time_unit)
        return time

    def get_speed(self, x, time_unit):
        """
        Calculate the number of events per unit time
        :param x: A Series of datetime values with one element per share event
        :param time_unit: Granularity of time (e.g. "D","m","h")
        :return: A scalar with number of total events over the total time elapsed
        """


        time = self.get_lifetime(x, time_unit)


        if time == 0:
            speed = np.nan
        else:
            speed = len(x) / time

        return speed


    def scalar_measurement(self, data, agg_col, agg_func, node_level=False, community_level=False, fill_missing=False):
        """
        Description: Calculate a scalar measurement

        Input:
            data: A DataFrame of event data
            agg_col: The field on which to perform the aggregation
            agg_func: The aggregation function to apply
            node_level: If true, computes order of spread of nodes passed or from metadate object
            community_level: If true, computes order of spread of nodes within each community
            If node_level and community_level both set to False, computes population level

        Output: If population level: a scalar which is the mean across the population of info IDs
                If community level : a dictionary mapping between each community and a scalar with the
                                      mean value per unit of information in that community
                If node level: a dictionary mapping between each node and a scalar value
        """

        
        group_cols = [self.content_col]
        if community_level:
            group_cols += [self.community_col]
            if self.community_col not in data:
                warnings.warn("No valid communities found in data")
                return None
            
        result = data.groupby(group_cols)[agg_col].apply(agg_func)
        result.name = 'value'
 
        if fill_missing:
            if not community_level:
                for node in self.node_list:
                    if node not in result.index:
                        result.loc[node] = 0.0
            else:
                to_add = []
                for community in self.community_dict.keys():
                    for node in self.community_dict[community]:
                        if node not in result.index:
                            to_add.append((node,community))
                for addition in to_add:
                    result.loc[addition] = 0.0

        if node_level:
            meas = result.replace({pd.np.nan: None}).to_dict()
        elif community_level:
            meas = result.reset_index().groupby(self.community_col).mean().replace({pd.np.nan: None}).to_dict()['value']
        else:
            meas = result.mean()
            if np.isnan(meas):
                meas = None

        return meas

    def distribution_measurement(self, data, agg_col, agg_func, extra_group_cols=[], 
                                 node_level=False, community_level=False, fill_missing=False):
        """
        Description: Determine the distribution of a property across different information IDs

        Input:
            data: A DataFrame of event data
            agg_col: The field on which to perform the aggregation
            agg_func: The aggregation function to apply
            extra_group_cols: A list of fields over which the distribution should be calculated in addition
                              to the information ID.  Used for distribution on the node_level.
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe with each info ID and the scalar aggregation value
                If community level : a dictionary mapping each community to a dataframe with
                                     each info ID in the community and its scalar aggregation value
                If node level: a dataframe with each value of the extra_group_cols and the scalar value
        """

        group_cols = [self.content_col] 
        if community_level:
            group_cols += [self.community_col]
            if self.community_col not in data:
                warnings.warn("No valid communities found in data")
                return None
            
        values = data.groupby(group_cols + extra_group_cols)[agg_col].apply(agg_func)
        values.name = 'value'
        
        if fill_missing:
            if not community_level:
                for node in self.node_list:
                    if node not in values.index:
                        values.loc[node] = 0.0
            else:
                to_add = []
                for community in self.community_dict.keys():
                    for node in self.community_dict[community]:
                        if node not in values.index:
                            addition = [node,community]
                            for col in extra_group_cols:
                                addition += ['']
                            to_add.append(tuple(addition))
                for addition in to_add:
                    values.loc[addition] = 0.0

        if len(extra_group_cols) > 0:
            values = values.reset_index().groupby(group_cols)#.mean()
            if not node_level:
                values = values.mean()

        if community_level:
            meas = values.reset_index()
            meas = dict(tuple(meas.groupby(self.community_col)))
            meas = {k:v[[self.content_col,'value']].dropna() for k,v in meas.items()}
        elif node_level:
            meas = dict(tuple(values))
            meas = {k:v[extra_group_cols + ['value']].dropna() for k,v in meas.items()}
        else:
            meas = values.reset_index().sort_values(by='value',ascending=False).dropna()

        return meas

    
    def topk_measurement(self, data, agg_col, agg_func, k=5, node_level=False, community_level=False):
        """
        Description: Determine the top K ranking of units of information based a specific aggregation

        Input:
            data: A DataFrame of event data
            agg_col: The field on which to perform the aggregation
            agg_func: The aggregation function to apply
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and their scalar values
                If community level : a dictionary maaping each community to a sorted dataframe
                                     with the top k nodes and their scalar values
                If node level: a sorted dataframe with the top k nodes and their counts.
                               Only differs from Population if subset of nodes is passed.
        """

        group_cols = [self.content_col]
        if community_level:
            group_cols += [self.community_col]
            if self.community_col not in data:
                warnings.warn("No valid communities found in data")
                return None

            
        values = data.groupby(group_cols)[agg_col].apply(agg_func)
        values.name = 'value'
        values = values.reset_index()

        if community_level:
            meas = values.groupby(self.community_col).apply(lambda x: x.nlargest(k,"value")).reset_index(drop=True)
            meas = dict(tuple(meas.groupby(self.community_col)))
            meas = {k:v[[self.content_col,'value']].reset_index(drop=True) for k,v in meas.items()}
        else:
            meas = values.nlargest(k,"value")
       
        return meas

    def temporal_measurement(self, data, agg_col, agg_func, time_bin='D',delta_t = False, 
                             node_level=False, community_level=False, fill_missing=False):
        """
        Description: Calculate a time series of quantity of interest

        Input:
            data: A DataFrame of event data
            agg_col: The field on which to perform the aggregation
            agg_func: The aggregation function to apply
            time_bin: Temporal granularity for the time series output
            node_level: If true, computes order of spread of nodes passed or from metadata object
            community_level: If true, computes order of spread of nodes within each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata
            If node_level and community_level both set to False, computes population level

        Output: If population level: a series with the average aggregation value at each time
                If community level : a dictionary mapping each community to a series with the
                                     average aggregation value at each time
                If node level: a dataframe of every node and their aggregation value at each time
        """
        
        group_cols = [self.content_col]
        if community_level:
            group_cols += [self.community_col]
            if self.community_col not in data:
                warnings.warn("No valid communities found in data")
                return None
            
        if agg_col == self.timestamp_col:
            data.loc[:,'agg_col'] = data.loc[:,self.timestamp_col]
            agg_col = 'agg_col'

        if len(data) == 0:
            meas = {}
            for node in self.node_list:
                meas[node] = pd.DataFrame(columns = ['nodeTime','value'])
            return meas
        
        
        if not delta_t:
            #get time series in absolute time
            data = data.set_index(self.timestamp_col)
            data = data.groupby([pd.Grouper(freq=time_bin)] + group_cols)[agg_col].apply(agg_func).reset_index()
        else:
            #get time series as function of time elapsed since first event
            grouped = data.groupby(group_cols)[self.timestamp_col].min().reset_index()
            grouped.columns = group_cols + ['first_event_time']
            data = data.merge(grouped,on=group_cols)
            data[self.timestamp_col] = data[self.timestamp_col] - data['first_event_time']

            if time_bin == 'D':
                data[self.timestamp_col] = data[self.timestamp_col].dt.days
            elif time_bin.lower() == 'h':
                data[self.timestamp_col] = data[self.timestamp_col].dt.hours
            elif time_bin.lower() == 'm':
                data[self.timestamp_col] = data[self.timestamp_col].dt.minutes
                
            data = data.groupby([self.timestamp_col] + group_cols)[agg_col].apply(agg_func).reset_index()

        if not community_level:
            data = pd.pivot_table(data,index=self.timestamp_col,columns=self.content_col,values=agg_col).fillna(0)
        else:

            if fill_missing:
                to_add = []
                for community in self.community_dict.keys():
                    for node in self.community_dict[community]:
                        if node not in data[self.content_col].values:
                            for time in data[self.timestamp_col].unique():
                                to_add.append([time,node,community,0.0])
                if len(to_add):
                    to_add = pd.DataFrame(to_add)
                    to_add.columns = [self.timestamp_col,self.content_col,self.community_col,agg_col]
                    data = pd.concat([data,to_add])


            def pivot_group(grp):

                grp = pd.pivot_table(grp,
                                      index=self.timestamp_col,
                                      columns=self.content_col,
                                      values=agg_col).fillna(0).mean(axis=1)
                return grp

            data = data.groupby(self.community_col).apply(lambda x: pivot_group(x))
 
           
            if 'nodeTime' not in data.index.names:
                #when there are two or less time bins the structure of the DF comes out wrong
                #fixing it here
                data = pd.melt(data.reset_index(),id_vars=self.community_col)
                data = data.set_index([self.community_col,self.timestamp_col])

            data.name = 'value'
            data = data.reset_index()

        if fill_missing:
            if not community_level:
                for node in self.node_list:
                    if node not in data.columns:
                        data[node] = np.zeros(len(data))

                        
        if node_level:
            meas = {col:pd.DataFrame(data[col]).rename(columns={col:'value'}).reset_index() for col in data.columns}
        elif community_level:
            meas = dict(tuple(data.groupby(self.community_col)))
            meas = {k:v[[self.timestamp_col,'value']].reset_index(drop=True) for k,v in meas.items()}
        else:
            data = data.mean(axis=1)
            data.name = 'value'
            meas = data.reset_index()
        
        return(meas)



    def number_of_shares(self, node_level=False, community_level=False, nodes=[], 
                         communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: number_of_shares (population_number_of_shares, community_number_of_shares, node_number of shares)

        Description: Determine the number of times a piece of information is shared

        Input:
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level
            platform: The platform to subset the data or "all" for all platforms

        Output: If population level: a scalar which is the mean number of shares per unit of information 
                                      in the population
                If community level : a dictionary mapping between each community and a scalar with the
                                      mean number of shares per unit of information in that community
                If node level: a dictionary mapping between each node and a scalar number of shares
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        num_shares = self.scalar_measurement(data, self.id_col, self.get_shares,
                                             node_level=node_level, 
                                             community_level=community_level,
                                             fill_missing=True)

        return num_shares


    def number_of_shares_over_time(self, time_bin='D',delta_t = False,
                                   node_level=False, community_level=False,
                                   nodes=[], communities=[], platform="all", action_types=[],
                                   date_range=[]):
        """
        Measurement: number_of_shares_over_time (population_number_of_shares_over_time, community_number_of_shares_over_time, population_number_of_shares_over_time)

        Description: Determine the number of times a piece of information is shared over time

        Input:
            time_bin: The time granularity of the time series
            delta_t: A boolean. If False, then calculate the time series based on absolute time (i.e. the number
                     of shares on each date). If True, calculate based on time elapsed since the first share (i.e.,
                     the number of shares N days since first share)
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level
            platform: The platform to subset the data or "all" for all platforms

        Output: If population level: a series with the average number of shares at each time
                If community level : a dictionary mapping each community to a series with the
                                     average number of shares at each time
                If node level: a dictionary mapping each node to a dataframe of counts at each time
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types,date_range)

        shares_time_series = self.temporal_measurement(data, self.id_col, self.get_shares, 
                                                       node_level=node_level, 
                                                       community_level = community_level,
                                                       delta_t=delta_t,time_bin=time_bin,
                                                       fill_missing=True)

        return shares_time_series


    def distribution_of_shares(self, node_level=False, community_level=False, 
                               nodes=[], communities=[], platform="all", action_types=[],
                               date_range=[]):
        """
        Measurement: distribution_of_shares (population_distribution_of_shares, community_distribution_of_shares, node_distribution_of_shares)

        Description: Determine the distribution of the amount of shares of information

        Input:
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe with each info ID and number of shares
                If community level : a dictionary mapping each community to a dataframe with
                                     each info ID in the community and its number of shares
                If node level: a dataframe with each info ID and number of shares
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform,action_types,date_range)

        shares_distribution = self.distribution_measurement(data, self.id_col, self.get_shares, 
                                                            node_level=node_level, 
                                                            community_level = community_level,
                                                            fill_missing=True)

        return shares_distribution


    def top_info_shared(self, k=5, node_level=False, community_level=False, 
                        nodes=[], communities=[], platform="all", action_types=[],
                        date_range=[]):
        """
        Measurement: top_info_shared (population_top_info_shared, community_top_info_shared, node_top_info_shared)

        Description: Determine the top K pieces of information shared the most.

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and their counts
                If community level : a dictionary mapping each community to a sorted dataframe
                                     with the top k nodes and their counts
                If node level: a sorted dataframe with the top k nodes and their counts.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        shares_topk = self.topk_measurement(data, self.id_col, self.get_shares, 
                                            node_level=node_level, community_level = community_level,
                                            k=k)

        return shares_topk

    def unique_users(self, node_level=False, community_level=False, 
                     nodes=[], communities=[], platform="all", action_types=[],
                     date_range = []):
        """
        Measurement: unique_users (population_unique_users, community_unique_users, node_unique_users)        

        Description: Calculate the number of unique users (audience) reached by each piece of information

        Input:
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a scalar with the mean number of unique users per info ID.
                If community level: a dictionary mapping each community to the mean number of unique users for info IDs
                                    in that community.
                If node level: a dictionary with the number of unique users for each info ID
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        num_unique_users = self.scalar_measurement(data, self.user_col, self.get_audience,
                                                   community_level=community_level,
                                                   node_level=node_level,
                                                   fill_missing=True)

        return num_unique_users

    def unique_users_over_time(self, time_bin='D',delta_t = False,
                               node_level=False, community_level=False,
                               nodes=[], communities=[], platform="all", action_types=[],
                               date_range=[]):
        """
        Measurement: uniqe_users_over_time (population_unique_users_over_time, community_unique_users_over_time, node_unique_users_over_time)

        Description: Calculate the number of users at each time step

        Input:
            time_bin: The time granularity of the time series
            delta_t: A boolean. If False, then calculate the time series based on absolute time (i.e. the number
                     of users on each date). If True, calculate based on time elapsed since the first share (i.e.,
                     the number of users N days since first share)
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a time series with the mean number of users at each time step
                If community level: a dictionary mapping each community to a time series with the mean number of users
                                    at each time step
                If node level: a dataframe where each node is a row and the columns contain the number of unique users
                               at each time step
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform,action_types,date_range)


        unique_users_time_series = self.temporal_measurement(data, self.user_col, self.get_audience,
                                                             community_level=community_level,
                                                             node_level=node_level,
                                                             time_bin=time_bin, delta_t=delta_t,
                                                             fill_missing=True)

        return unique_users_time_series



    def distribution_of_users(self, node_level=False, community_level=False,
                              nodes=[], communities=[], platform="all", action_types=[],
                              date_range=[]):
        """
        Measurement: distribution_of_users (population_distribution_of_users, community_distribution_of_users, node_distribution_of_users)

        Description: Determine the distribution of the number of users reached

        Input:
            If node_level and community_level both set to False, computes population level
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            

        Output: If population level: a dataframe of the number of unique users and their counts.
                If community level : a dictionary mapping each community to a dataframe of the number of unique users
                                     and their counts
                If node level: a dataframe of the number of unique users and their counts.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        unique_users_distribution = self.distribution_measurement(data, self.user_col, self.get_audience,
                                                                  community_level=community_level,
                                                                  node_level=node_level,
                                                                  fill_missing=True)
                                                       
        return unique_users_distribution



    def top_audience_reach(self, k, node_level=False, community_level=False,
                           nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: top_audience_reach (population_top_audience_reach, community_top_audiene_reach, node_top_audience_reach)

        Description: Determine the top K pieces of information with the largest audience reach.

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and the size of audience reached
                If community level : a dictionary maaping each community to a sorted dataframe
                                     with the top k nodes and the size of audience reached
                If node level: a sorted dataframe with the top k nodes and the size of audience reached.
                                Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        unique_users_topk = self.topk_measurement(data, self.user_col, self.get_audience,
                                                  community_level=community_level,
                                                  node_level=node_level,
                                                  k=k)
                                                       
        return unique_users_topk


    def lifetime_of_info(self, time_unit='D',
                         node_level=False, community_level=False, 
                         nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: lifetime_of_info (population_lifetime_of_info, community_lifetime_of_info, node_lifetime_of_info)

        Description: Calculate the lifetime (default = days) of the pieces of information

        Input:
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a scalar with the mean lifetime of the info IDs
                If community level: a dictionary mapping each community to a the mean lifetime of info IDs
                                    in that community
                If node level: a dictionary mapping each node to its lifetime
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        lifetime = self.scalar_measurement(data, self.timestamp_col, lambda x: self.get_lifetime(x, time_unit),
                                           community_level=community_level,
                                           node_level=node_level,
                                           fill_missing=True)
        
        return lifetime


    def lifetime_of_info_distribution(self, time_unit='D',
                                      node_level=False, community_level=False, 
                                      nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: distribution_of_lifetimes (population_distribution_of_lifetimes, community_distribution_of_lifetimes, node_distribution_of_lifetimes)

        Description: Calculate the lifetime (default = days) of the pieces of information

        Input:
            time_unit: Temporal unit of lifetime
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a scalar with the mean lifetime of the info IDs
                If community level: a dictionary mapping each community to a the mean lifetime of info IDs
                                    in that community
                If node level: a dictionary mapping each node to its lifetime
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        lifetime = self.distribution_measurement(data, self.timestamp_col, lambda x: self.get_lifetime(x, time_unit),
                                                 community_level=community_level,
                                                 node_level=node_level,
                                                 fill_missing=True)

        return lifetime



    def lifetime_of_threads(self, time_unit = 'D', node_level=False, community_level=False, 
                            nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: lifetime_of_threads (population_lifetime_of_threads, community_lifetime_of_threads, node_lifetime_of_threads)

        Description: Calculate the lifetime (default = days) of shares/interactions with the initial shares (roots) of infoIDs

        Input:
            time_unit: Temporal unit of lifetime
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a distribution of the average lifetime of a root share for each piece of info
                If community level: a dictionary mapping each community to a distribution of
                                    the average lifetime of a root share for each piece of info
                                    in the community
                If node level: a dictionary of nodes with a dataframe of all root shares and their lifetimes for the
                               node.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        data.loc[data[self.root_col].isna(),self.root_col] = data.loc[data[self.root_col].isna(),self.id_col]

        fill_missing=True
        if node_level:
            fill_missing=False


        lifetime = self.distribution_measurement(data, self.timestamp_col, lambda x: self.get_lifetime(x, time_unit),
                                                 extra_group_cols = [self.root_col],
                                                 community_level=community_level,
                                                 node_level=node_level,
                                                 fill_missing=fill_missing)
        
        return lifetime


    def distribution_of_lifetimes(self, time_unit="D", node_level=False, community_level=False,
                                  nodes=[], communities=[], platform="all", action_types=[],
                                  date_range=[]):
        """
        Description: Calculate the distributions of the information lifetimes

        Input:
            time_unit: Granularity of time
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the lifetimes of nodes and their counts
                If community level: a dictionary mapping each community to a dataframe of the
                                    lifetimes of nodes and their counts
                If node level: a dataframe of the lifetimes of nodes and their counts.
                                Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        speed_distribution = self.distribution_measurement(data, self.timestamp_col, lambda x: self.get_lifetime(x, time_unit),
                                                           community_level=community_level,
                                                           node_level=node_level,
                                                           fill_missing=True)
        return speed_distribution



    def top_lifetimes(self, time_unit='D', k=5, node_level=False, community_level=False, 
                      nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: top_lifetimes (population_top_lifetimes, community_top_lifetimes, node_top_lifetimes)

        Description: Determine the top K pieces of information with the longest lifetimes

        Input:
            k: Int. Specifies the number of pieces of information ranked
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a sorted dataframe with the top k nodes and lifetimes
                If community level : a dictionary mapping each community to a sorted dataframe
                                     with the top k nodes and lifetimes
                If node level: a sorted dataframe with the top k nodes and lifetimes.
                               Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)

        lifetime_topk = self.topk_measurement(data, self.timestamp_col, lambda x: self.get_lifetime(x, time_unit),
                                              community_level=community_level,
                                              node_level=node_level,
                                              k=k)
        
        return lifetime_topk


    def speed_of_info(self, time_unit="D", node_level=False, community_level=False,
                      nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: speed_of_info (population_speed_of_info, community_speed_of_info, node_speed_of_info)

        Description: Determine the speed at which each piece of information is spreading

        Input:
            time_unit: Granularity of time
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a scalar value with the avarage speed of info IDs in the population
                If community level : a dictionary mapping each community to the mean speed of info IDs
                                     in that community
                If node level: a dictionary of nodes and their speeds
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)


        speed = self.scalar_measurement(data, self.timestamp_col, lambda x: self.get_speed(x, time_unit),
                                        community_level=community_level,
                                        node_level=node_level,fill_missing=True)
        return speed


    def speed_of_info_over_time(self, time_unit="D", time_bin='D', delta_t = False,
                                node_level=False, community_level=False,
                                nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: speed_of_info_over_time (population_speed_of_info_over_time, community_speed_of_info_over_time, node_speed_of_info_over_time)
        Description: Determine the speed (default = shares/day) of information over time

        Input:
            time_unit: Granularity of time for speed value (i.e. share per day or shares per hour, etc)
            time_bin: Granularity of time for time series
            delta_t: A boolean. If False, then calculate the time series based on absolute time (i.e. the speed
                     on each date). If True, calculate based on time elapsed since the first share (i.e.,
                     the speed N days since first share)
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the average speed at each time step of all nodes
                If community level : a dictionary mapping each community to a dataframe of the average speed at each
                                    time step of every node in the community
                If node level: a dictionary mapping each node to a dataframe of speeds at each time step
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)


        speed_time_series = self.temporal_measurement(data, self.timestamp_col, lambda x: self.get_speed(x, time_unit),
                                                      community_level=community_level,
                                                      node_level=node_level,
                                                      time_bin=time_bin, delta_t=delta_t,fill_missing=True)
        
        return speed_time_series


    def distribution_of_speed(self, time_unit="D", node_level=False, community_level=False,
                              nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: distribution_of_speed (population_distribution_of_speed, community_distribution_of_speed, node_distribution_of_speed)

        Description: Calculate the distributions of the varying speeds (default = shares/day) of the information

        Input:
            time_unit: Granularity of time
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: a dataframe of the speeds of nodes and their counts
                If community level: a dictionary mapping each community to a dataframe of the
                                    speeds of nodes and their counts
                If node level: a dataframe of the speeds of nodes and their counts.
                                Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)


        speed_distribution = self.distribution_measurement(data, self.timestamp_col, lambda x: self.get_speed(x, time_unit),
                                                           community_level=community_level,
                                                           node_level=node_level,
                                                           fill_missing=True)
        
        return speed_distribution


    def top_speeds(self, k, time_unit="D", node_level=False, community_level=False,
                   nodes=[], communities=[], platform="all", action_types=[], date_range=[]):
        """
        Measurement: top_speeds (population_top_speeds, community_top_speeds, node_top_speeds)

        Description: Determine the top K pieces of information that spread the fastest.

        Input:
            k: Int. Specifies the number of pieces of information ranked
            time_unit: Granularity of time (e.g. shares per day or shares per week)
            node_level: If true, compute measurement for each node
            community_level: If true, compute measurement for each community
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            communities: List of specific communities, keyword "all" or empty list (default) to use communities
                    provided from metadata if community_level is true
            If node_level and community_level both set to False, computes population level

        Output: If population level: dataframe with the top k nodes ranked descending order by speed.
                                     Position in dataframe is rank.
                If community level : Dictionary mapping communities to ranked dataframes of top k nodes
                                     in that community.
                If node level: dataframe with the top k nodes ranked descending order by speed. Position
                                in dataframe is rank. Only differs from Population if subset of nodes is passed.
        """
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types, date_range)


        speed_topk = self.topk_measurement(data, self.timestamp_col, lambda x: self.get_speed(x, time_unit),
                                           community_level=community_level,
                                           node_level=node_level,
                                           k=k)
        
        return speed_topk
    
    
    def activated_users(self,ratio=False, node_level=False,
                        nodes=[], platform="all", action_types=[],fill_missing=True):
        """
        Measurement: activated_users 
        Description: Calculate the total number of new users that have not been active previously

        Input:
            ratio: If true, compute the ratio of new users to total unique users
            node_level: If true, compute measurement for each node
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            If node_level is set to False, computes population level

        Output: If population level: scalar which is the total number of activated users
                If node level: a dictionary where keys indicate the node and the values contain the total number of activated users
        """
        community_level = False
        communities=[]
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types)
        previous = self.previous_user_data
        if ratio:
            total_unique = data[self.user_col].nunique()
            node_unique = data.groupby(self.content_col)[self.user_col].apply(lambda x: x.nunique())
            
        if node_level:
            meas = {}
            for content, content_df in data.groupby(self.content_col):
                meas[content] = len(set(content_df[self.user_col])-set(previous[previous[self.content_col]==content][self.user_col]))
                if ratio:
                    meas[content] = meas[content]/node_unique.loc[content]


            if fill_missing:
                for node in self.node_list:
                    if node not in meas.keys():
                        meas[node] = 0.0
                        
        else:
            meas = len((set(data[self.user_col])-set(previous[self.user_col])))
            
            if ratio: 
                meas = meas/total_unique
                
        return meas
    
    
    def activated_users_over_time(self,time_bin="D",ratio=False,
                                  node_level=False,
                                  nodes=[], platform="all", action_types=[],fill_missing=True):
        """
        Measurement: activated_users_over_time

        Description: Calculate the number of new users at each time step

        Input:
            time_bin: The time granularity of the time series
            ratio: If true, compute the ratio of new users to total unique users
            node_level: If true, compute measurement for each node
            nodes: List of specific nodes to calculate measurement, or keyword "all" to calculate on all nodes, or
                    empty list (default) to calculate nodes provided in metadata if node_level is true
            If node_level is set to False, computes population level

        Output: If population level: a time series with total activated users at each time step
                If node level: a dataframe where each node is a row and the columns contain the number of activated users
                               at each time step
        """
        community_level = False
        communities=[]
        data = self.preprocess(node_level, nodes, community_level, communities, platform, action_types)
        
        #get rid of unnecessary columns
        data = data[[self.timestamp_col,self.user_col,self.content_col]]
        
        #set previous nodetime earlier than all new data
        previous_users = self.previous_user_data
        previous_users[self.timestamp_col]= data[self.timestamp_col].min()-pd.Timedelta("2 {}".format(time_bin))

        #set a value to be able to remove previous events later
        previous_users["is_previous"]=True
        data["is_previous"]=False

        data = pd.concat([previous_users,data],sort=True)
        data = data.sort_values(self.timestamp_col)

        if node_level:
            if ratio:
                unique_users = data[~data.is_previous].set_index(self.timestamp_col).groupby([self.content_col,pd.Grouper(freq=time_bin)])[self.user_col].apply(lambda x: x.nunique())
                unique_users = unique_users.reset_index(0).pivot(columns="informationID")["nodeUserID"]
                
            data = data.drop_duplicates(subset=[self.content_col,self.user_col],keep="first")
            data = data[~data.is_previous]
            if len(data)==0:
                meas = {}
                for node in self.node_list:
                    if node not in data.columns:
                        meas[node] = pd.DataFrame(columns=[self.timestamp_col,'value'])
            else:
                data = data.set_index(self.timestamp_col).groupby([self.content_col,pd.Grouper(freq=time_bin)]).apply(len)
                data = data.reset_index(0).pivot(columns=self.content_col)[0]
                if fill_missing:
                    for node in self.node_list:
                        if node not in list(data.columns):
                            data[node] = np.zeros(len(data))
                            
                if ratio:
                    data = data/unique_users
                meas = {col:pd.DataFrame(data[col]).rename(columns={col:'value'}).reset_index().fillna(0) for col in data.columns}

        else:
            if ratio:
                unique_users = data[~data.is_previous].set_index(self.timestamp_col).groupby([pd.Grouper(freq=time_bin)])[self.user_col].apply(lambda x: x.nunique())
                
            data = data.drop_duplicates(subset=[self.user_col],keep="first")
            data = data[~data.is_previous]
            data = data.set_index(self.timestamp_col).groupby(pd.Grouper(freq=time_bin)).apply(len)
            if ratio:
                data = data/unique_users
            meas = pd.DataFrame(data).reset_index().rename({0:"value"},axis=1).fillna(0)

        return meas
