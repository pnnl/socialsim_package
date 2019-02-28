import pandas as pd
import pickle as pkl
import numpy  as np

from .measurements import MeasurementsBaseClass
from .validators   import check_empty
from .validators   import check_root_only

"""
Notes:
    - If no metadata then run community measurements as if there is a single 
    community containing every node.
"""

class SocialActivityMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, metadata, platform,
        content_node_ids=[], user_node_ids=[]):
        """
        Description:

        Input:
            :dataset:
            :configuration:
            :metadata:

        Output:
            None
        """
        super().__init__(dataset, configuration)

        self.measurement_type = 'social_activity'
        self.platform         = platform


        self.contribution_events = [
            'PullRequestEvent',
            'PushEvent',
            'IssuesEvent',
            'IssueCommentEvent',
            'PullRequestReviewCommentEvent',
            'CommitCommentEvent',
            'CreateEvent',
            'post',
            'tweet'
            ]

        self.popularity_events = [
            'WatchEvent',
            'ForkEvent',
            'comment',
            'post',
            'retweet',
            'quote',
            'reply'
            ]

        self.main_df = self.preprocess(dataset)

        # store action and merged columns in a seperate data frame that is not 
        # used for most measurements
        if platform=='github' and len(self.main_df.columns)==6 and 'action' in self.main_df.columns:
            self.main_df_opt = self.main_df.copy()[['action', 'merged']]
            self.main_df = self.main_df.drop(['action', 'merged'], axis=1)
        else:
            self.main_df_opt = None

        # For content centric
        if content_node_ids=='all':
            self.selectedContent = self.main_df
        else:
            if self.platform in ['reddit', 'twitter']:
                self.selectedContent = self.main_df[self.main_df.root.isin(content_node_ids)]
            elif self.platform=='github':
                self.selectedContent = self.main_df[self.main_df.content.isin(content_node_ids)]

        # For userCentric
        self.selectedUsers = self.main_df[self.main_df.user.isin(user_node_ids)]

        if metadata:
            if metadata.use_content_data:
                self.useContentMetaData = True
                self.contentMetaData    = metadata.content_data

            if metadata.use_user_data:
                self.useUserMetaData = True
                self.UserMetaData    = metadata.user_data
        else:
            self.useContentMetaData = False
            self.useUserMetaData    = False

        # For community measurements
        # Load and preprocess metadata
        if self.useUserMetaData and self.useContentMetaData:
            self.comDic = metadata.build_communities(self.contentMetaData,
                self.UserMetaData)
        else:
            self.comDic = {}

        if self.platform=='github':
            self.communityDF = self.getCommmunityDF('community')
        elif self.platform=='reddit':
            self.communityDF = self.getCommmunityDF('subreddit')
        elif self.platform=='twitter':
            self.communityDF = self.getCommmunityDF('')


    def preprocess(self, dataset):
        """
        Description:

        Input:

        Output:
        Edit columns, convert date, sort by date
        """
        events  = self.popularity_events+self.contribution_events
        mapping = {'actionType'   : 'event',
                   'nodeID'       : 'content',
                   'nodeTime'     : 'time',
                   'nodeUserID'   : 'user'}

        if self.platform=='reddit':
            mapping.update({'communityID' : 'subreddit',
                            'keywords'    : 'keywords',
                            'parentID'    : 'parent',
                            'rootID'      : 'root'})
        elif self.platform=='twitter':
            mapping.update({'parentID' : 'parent',
                            'rootID'   : 'root'})
        elif self.platform=='github':
            mapping.update({'actionSubType' : 'action',
                            'status'        : 'merged'})

        dataset = dataset.rename(index=str, columns=mapping)
        dataset = dataset[dataset.event.isin(events)]
        dataset = dataset.sort_values(by='time')
        dataset = dataset.assign(time=dataset.time.dt.floor('h'))

        return dataset


    def readPickleFile(self, filepath):
        with open(filepath, 'rb') as f:
            pickle_object = pkl.load(f)
        return pickle_object


    def getCommmunityDF(self, community_col):
        if community_col in self.main_df.columns:
            return self.main_df.copy()

        elif community_col!='':
            dfs = []

            content_community_types = ['topic','language']
            user_community_types    = ['city','country','company','locations']

            #content-focused communities
            for community in content_community_types:
                if community in self.comDic.keys():
                    for key in self.comDic[community]:
                        d = self.main_df[self.main_df['content'].isin(self.comDic[community][key])]
                        d[community_col] = key
                        dfs.append(d)

            #user-focused communities
            for community in user_community_types:
                if community in self.comDic.keys():
                    for key in self.comDic[community]:
                        d = self.main_df[self.main_df['user'].isin(self.comDic[community][key])]
                        d[community_col] = key
                        dfs.append(d)

            if len(dfs)==0:
                result = self.main_df.copy()
                result['community'] = 'all'
            else:
                result = pd.concat(dfs)

            return result


    def getCommunityMeasurementDict(self, dataset):
        measurements = {}
        if isinstance(dataset, pd.DataFrame):
            for community in dataset['community'].unique():
                measurements[community]=dataset[dataset.community==community]
                del measurements[community]['community']
        elif isinstance(dataset, pd.Series):
            series_output = False
            for community in dataset.index:
                measurements[community] = dataset[community]
                try:
                    len(dataset[community])
                    series_output = True
                except:
                    pass

            if series_output:
                for community in measurements:
                    measurements[community] = pd.Series(measurements[community])

        return measurements


    def getProportion(self, eventTypes=None, community_field="subreddit"):
        """
        Calculates the proportion of each event type in the data.
        Question #7
        Inputs: df - Events data frame
                communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
                eventTypes - List of event types to include
        Output: Dictionary of data frames with columns for event type and proportion, with one data frame for each community
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        p = df[['user','event',community_field]].groupby([community_field,'event']).count()
        p = p.reset_index()
        p.columns = ['community','event', 'value']

        community_totals = p[['community','value']].groupby('community').sum().reset_index()
        community_totals.columns = ['community','total']
        p = p.merge(community_totals, on='community',how='left')

        p['value'] = p['value']/p['total']
        del p['total']

        measurement = self.getCommunityMeasurementDict(p)

        return measurement


    def contributingUsers(self,eventTypes=None,community_field="subreddit"):
        """
        This method calculates the proportion of users with events in teh data who are active contributors.
        Question #20
        Inputs: df - Events data
        Output: Proportion of all users who make active contributions
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        #total number of unique users
        totalUsers = df.groupby(community_field)['user'].nunique()
        totalUsers.name = 'total_users'

        df = df[df['event'].isin(self.contribution_events)]

        #number of unique users with direct contributions
        contribUsers = df.groupby(community_field)['user'].nunique()
        contribUsers.name = 'contributing_users'

        df = pd.concat([totalUsers, contribUsers], axis=1).fillna(0)

        df['value'] = df['contributing_users']/ df['total_users']

        measurement = self.getCommunityMeasurementDict(df['value'])

        return measurement


    def getNumUserActions(self,unit='h',eventTypes=None,community_field='subreddit'):
        """
        Calculate the average temporal user contribution counts within the data set.
        Question #23
        Inputs: df - Events data frame
                unit - Time granularity for time series calculation
                eventTypes - List of event types to include
        Output: Data frame containing a time series of average event counts
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        df['value'] = [0 for i in range(len(df))]
        df = df.set_index('time')

        #get event counts for each user within each time unit
        df = df[['user','value',community_field]].groupby([ pd.TimeGrouper(unit), 'user', community_field]).count()
        df = df.reset_index()

        #average the event counts across all users to get a single time series for the community
        df = df[['time','value',community_field]].groupby(['time',community_field]).mean().reset_index()
        df['value'] = pd.to_numeric(df['value'])
        df.columns = ['time','community','value']

        measurement = self.getCommunityMeasurementDict(df)

        return measurement


    def burstsInCommunityEvents(self,eventTypes=None,community_field="subreddit"):
        """
        Calculates the burstiness of inter-event times within the data set.
        Question #9
        Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
                eventTypes - List of event types to include in the data
        Output: Burstiness value (scalar)
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        if len(df) == 0:
            return None

        def burstiness(grp):

            #get interevent times
            grp['diff'] = grp['time'].diff()
            grp['diff'] = grp['diff'] / np.timedelta64(1, 's')

            grp = grp[np.isfinite(grp['diff'])]

            mean = grp['diff'].mean()
            std = grp['diff'].std()
            if std + mean > 0:
                burstiness = (std - mean) / (std + mean)
            else:
                burstiness = 0

            return burstiness

        b = df.groupby(community_field).apply(burstiness)
        b.columns = ['community','value']

        measurement = self.getCommunityMeasurementDict(b)

        return measurement


    def propIssueEvent(self,unit='D'):
        """
        Calculates the proportion of different issue action types as a function of time.
        Question #8 (Optional Measurement)
        Inputs: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
                unit - Temporal granularity for calculating the time series (e.g. D - day, H - hour, etc.)
        Output: Dictionary of data frames for each community
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if self.main_df_opt is not None:

            df = df[ (df['event'] == 'IssuesEvent') ]

            if len(df) == 0:
                return(None)

            #round times down to nearest unit
            df = df.assign(time=df.time.dt.floor(unit))

            #merge optional columns (action, merged) with primary data frame
            df = df.merge(self.main_df_opt,how='left',left_index=True,right_index=True)

            df = df[(df['action'].isin(['opened','closed','reopened']))]

            if len(df) == 0:
                return(None)

            df = df[['action','event','time','community']].groupby(['time','action','community']).count()  #time,action,count
            df = df.reset_index()

            p = df
            p.columns = ['time','action', 'community','counts']

            #create one column for each action type holding the counts of that action type
            p = pd.pivot_table(p,index=['time','community'],columns='action', values='counts').fillna(0)
            p = p.reset_index()
            p = pd.melt(p, id_vars=['time','community'], value_vars=['closed', 'opened', 'reopened'])
            p.columns = ['time','community','action', 'value']

            measurement = self.getCommunityMeasurementDict(p)

            return measurement
        else:
            return None


    def ageOfAccounts(self,eventTypes=None,community_field="subreddit"):
        """
        Calculates the distribution of user account ages for users who are active in the data.
        Question #10
        Inputs: df - Events data
                     eventTypes - List of event types to include in the data
        Output: A pandas Series containing account age of the user for each action taken in the community
        """
        df = self.communityDF.copy()

        if len(df) == 0:
            return None

        if self.useUserMetaData:

            if eventTypes != None:
                df  = df[df.event.isin(eventTypes)]


            df = df.merge(self.created_at_df, left_on='user', right_on='user', how='inner')
            df = df.sort_values(['time'])

            #get user account age at the time of each event
            df['age'] = df['time'].sub(df['created_at'], axis=0)
            df['age'] = df['age'].astype('timedelta64[D]')

            df = df.rename(index=str, columns={community_field: "community"})
            df.set_index('community')

            measurement = self.getCommunityMeasurementDict(df['age'])

            return df['age']
        else:
            warnings.warn('Skipping ageOfAccountsHelper because metadata file is required')
            return None


    def userGeoLocation(self,eventTypes=None,community_field="subreddit"):
        """
        A function to calculate the distribution of user geolocations for users active in each community
        Question #21
        Inputs: df - Events data frame
                eventTypes - List of event types to include in the data
        Output: Data frame with the location distribution of activity in the data
        """
        df = self.communityDF.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        if len(df) == 0:
            return None

        if self.useUserMetaData:

            #merge events data with user location metadata
            merge = df.merge(self.locations_df, left_on='user', right_on='user', how='inner')
            merge = merge[['user','country',community_field]].groupby([community_field,'country']).count().reset_index()
            merge.columns = ['community','country','value']

            community_totals = merge.groupby(community_field)['value'].sum().reset_index()
            community_totals.columns = ['community','total']
            merge = merge.merge(community_totals,on='community',how='left')
            merge['value'] = merge['value']/merge['total']

            print('merge',merge)

            #set rare locations to "other"
            thresh = 0.007
            merge['country'][merge['value'] < thresh] = 'other'

            #sum over other countries
            grouped = merge.groupby([community_field,'country']).sum().reset_index()

            print('grouped',grouped)

            measurement = self.getCommunityMeasurementDict(grouped)

            return measurement
        else:
            warnings.warn('Skipping userGeoLocationHelper because metadata file is required')
            return {}


    def getUserBurstByCommunity(self,eventTypes=None,thresh=5.0,community_field="subreddit"):
        """
        Calculate the distribution of user inter-event time burstiness in the data set.
        Question #9
        Inputs: df - Events data frame
                eventTypes - List of event types to include in the data
                thresh - Minimum number of events for a user to be included
        Output: Data frame of user burstiness values
        """
        df = self.communityDF.copy()

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        if len(df) == 0:
            return None

        #only calculate burstiness for users which have sufficient activity
        users = df.groupby(['user',community_field])
        user_counts = users['event'].count().reset_index()
        user_list = user_counts[user_counts['event'] >= thresh]
        user_list.columns = ['user',community_field,'total_activity']

        if len(user_list) == 0:
            return None

        df = df.merge(user_list,how='inner',on=['user',community_field])

        def user_burstiness(grp):
            #get interevent times for each user seperately
            if len(grp['user'].unique()) > 1:
                grp = grp.groupby('user').apply(lambda grp: (grp.time - grp.time.shift()).fillna(0)).reset_index()
            else:
                grp['time'] = grp['time'] - grp['time'].shift().dropna()

            grp['value'] = grp['time'] / np.timedelta64(1, 's')

            #calculate burstiness using mean and standard deviation of interevent times
            grp = grp.groupby('user').agg({'value':{'std':np.std,'mean':np.mean}})

            grp.columns = grp.columns.get_level_values(1)
            grp['value'] = (grp['std'] - grp['mean']) / (grp['std'] + grp['mean'])

            grp = grp[['value']].dropna().reset_index()

            return grp

        b = df.groupby(community_field).apply(user_burstiness).reset_index()[[community_field,'value']].set_index(community_field)['value']

        measurement = self.getCommunityMeasurementDict(b)

        return measurement


    def getCommunityGini(self,communities=True,eventTypes=None,community_field="subreddit",content_field="root"):
        """
        Wrapper function calculate the gini coefficient for the data frame.
        Question #6
        Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
               eventTypes - A list of event types to include in the calculation
        Output: A dictionary of gini coefficients for each community
        """
        if len(self.communityDF) > 0:
            ginis = self.communityDF.groupby(community_field).apply(lambda x: self.getGiniCoefHelper(x,content_field))

            measurement = self.getCommunityMeasurementDict(ginis)

            return measurement
        else:
            return None


    def getCommunityPalma(self,communities=True,eventTypes=None,community_field="subreddit",content_field="root"):
        """
        Wrapper function calculate the Palma coefficient for the data frame.
        Question #6
        Input: communities - Boolean to calculate the measurement seperately for each community (True) or for the full data set (False)
               eventTypes - A list of event types to include in the calculation
        Output: A dictionary of Palma coefficients for each community
        """
        if len(self.communityDF) > 0:

            palmas = self.communityDF.groupby(community_field).apply(lambda x: self.getPalmaCoefHelper(x,content_field))

            measurement = self.getCommunityMeasurementDict(palmas)

            return measurement

        else:
            return None


    def getNodeDictionary(self,df):
        meas = {}
        for content in df.content.unique():
            meas[content] = df[df.content == content]
            del meas[content]["content"]

        return meas


    def getSelectContentIds(self, content_ids):
        """
        This function creates a dictionary of data frames with
        each entry being the activity of one piece of content from the content_ids
        argument.

        This is used for the selected content ids for the node-level meausurements.
        Inputs: content_ids - List of content ids (e.g. GitHub - full_name_h, etc.)
        Output: Dictionary of data frames with the content ids as the keys
        """
        contentDic = {}
        for ele in content_ids:
            d = self.main_df[self.main_df['content'] == ele]
            contentDic[ele] = d

        return contentDic


    def runSelectContentIds(self, method, *args):
        """
        This function runs a particular measurement (method) on the
        content ids that were selected by getSelectContentIds.

        This is used for the selected content IDs for the node-level meausurements.

        Inputs: method - Measurement function
        Output: Dictionary of measurement results with the content ids as the keys
        """
        ans = {}
        for ele in self.selectedContent.keys():
            df = self.selectedContent[ele].copy()
            ans[ele] = method(df,*args)

        return ans


    def getContentDiffusionDelay(self, eventTypes=None, selectedContent=True, time_bin='m',content_field='root'):
        """
        This method returns the distributon for the diffusion delay for each content node.
        Question #1
        Inputs: DataFrame - Data
            eventTypes - A list of events to filter data on
            selectedContent - A boolean indicating whether to run on selected content nodes
            time_bin - Time unit for time differences, e.g. "s","d","h"
        Output: An dictionary with a data frame for each content ID containing the diffusion delay values in the given units
        """
        df = self.selectedContent.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        if len(df.index) == 0:
            return {}

        #use metadata for content creation dates if available
        if self.useContentMetaData:
            df = df.merge(self.contentMetaData,left_on=content_field,right_on=content_field,how='left')
            df = df[[content_field,'created_at','time']].dropna()
            df['value'] = (df['time']-df['created_at']).apply(lambda x: int(x / np.timedelta64(1, time_bin)))
        #otherwise use first observed activity as a proxy
        else:
            creation_day = df.groupby(content_field)['time'].min().reset_index()
            creation_day.columns = [content_field,'creation_date']
            df = df.merge(creation_day, on=content_field, how='left')
            df['value'] = (df['time']-df['creation_date']).apply(lambda x: int(x / np.timedelta64(1, time_bin)))
            df = df[[content_field,'value']]
            df.columns = ['content','value']
            df = df.iloc[1:]

        measurements = self.getNodeDictionary(df)

        return measurements


    def getContentGrowth(self, eventTypes=None, cumSum=False, time_bin='D', content_field='root'):
        """
        This method returns the growth of a repo over time.
        Question #2
        Input:   eventTypes - A list of events to filter data on
                 cumSum - This is a boolean that indicates if the dataframe should be cumuluative over time.
                 time_bin - The temporal granularity of the output time series
        output - A dictionary with a dataframe for each content id that describes the content activity growth.
        """
        df = self.selectedContent

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]

        df = df.set_index("time")

        measurement = df[[content_field,'event']].groupby([content_field,pd.Grouper(freq=time_bin)]).count()
        measurement.columns = ['value']

        if cumSum == True:
            measurement['value'] = measurement.cumsum(axis=0)['value']
        measurement = measurement.reset_index()
        measurement.columns = ['content','time','value']

        measurements = self.getNodeDictionary(measurement)

        return measurements


    def getContributions(self, new_users_flag=False, cumulative=False, eventTypes=None, time_bin='H', content_field="root"):
        """
        Calculates the total number of unique daily contributers to a repo or the unique daily contributors who are new contributors
        Question # 4
            Input: newUsersOnly - Boolean to indicate whether to calculate total daily unique users (False) or daily new contributers (True),
                                  if None run both total and new unique users.
                   cumulative - Boolean to indicate whether or not the metric should be cumulative over time
                   eventTypes - A list of event types to include in the calculation
                   time_bin - Granularity of time series
            Output: A data frame with daily event counts
        """
        df = self.selectedContent.copy()

        def contributionsInsideHelper(dfH,newUsersOnly,cumulative):
            if newUsersOnly:
                #drop duplicates on user so a new user only shows up once in the data
                dfH = dfH.drop_duplicates(subset=['user'])

            p = dfH[[content_field,'user']].groupby([content_field,pd.Grouper(freq=time_bin)])['user'].nunique().reset_index()

            if cumulative:
                #get cumulative user counts
                p['user'] = p.groupby(content_field)['user'].transform(pd.Series.cumsum)

            p.columns = ['content','time','value']
            return p

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        df = df.set_index("time")

        if not new_users_flag:
            #run total daily user counts
            results = contributionsInsideHelper(df,False, cumulative)
        else:
            #run unique daily user counts
            results = contributionsInsideHelper(df,newUsersOnly, cumulative)

        meas = self.getNodeDictionary(results)

        return meas


    def getDistributionOfEvents(self,weekday=False,content_field="root"):
        """
        This method returns the distribution for each event over time or by weekday. Default is over time.
        Question #5
        Inputs: weekday - (Optional) Boolean to indicate whether the distribution should be done by weekday. Default is False.
        Output: Dataframe with the distribution of events by weekday. Columns: Event, Weekday, Count or Event, Date, Count
        """
        df = self.selectedContent.copy()

        df['id'] = df.index
        df['weekday'] = df['time'].dt.weekday_name
        df['date'] = df['time'].dt.date

        if weekday:
            col = 'weekday'
        else:
            col = 'date'

        counts = df.groupby([content_field,'event',col])['user'].count().reset_index()
        counts.columns = ['content','event',col,'value']

        meas = self.getNodeDictionary(counts)

        return meas


    def processDistOfEvents(self,df,weekday):
        """
        Helper Function for getting the Dist. of Events per weekday.
        """
        df.set_index('time', inplace=True)
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year

        if weekday:
            df['weekday'] = df.apply(lambda x:datetime(x['year'],x['month'],x['day']).weekday(),axis=1)
            p = df[['event','user','weekday']].groupby(['event','weekday']).count()
            p = p.reset_index()
            return p

        else:
            p = df[['event', 'year', 'month', 'day','id']].groupby(['event', 'year', 'month','day']).count()
            p = pd.DataFrame(p).reset_index()
            p.column = ['event', 'year', 'month','day','count']
            p['date'] = p.apply(lambda x: datetime.strptime("{0} {1} {2}".format(x['year'], x['month'],x['day']), "%Y %m %d"), axis=1)
            p['date'] = p['date'].dt.strftime('%Y-%m-%d')
            p = p.reset_index()
            return p


    @check_empty(default=None)
    def getGiniCoef(self,nodeType='root', eventTypes=None, 
        content_field="root"):
        """
        Wrapper function calculate the gini coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: g - gini coefficient
        """
        result = self.getGiniCoefHelper(self.main_df, nodeType, eventTypes, content_field)

        return result


    def getGiniCoefHelper(self, df, nodeType, eventTypes=None, 
        content_field="root"):
        """
        This method returns the gini coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Gini coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: g - gini coefficient
        """
        if eventTypes is not None:
            df = df[df.event.isin(eventTypes)]

        if len(df) == 0:
            return None

        #count events for given node type
        if nodeType != 'user':
            df = df[[nodeType, 'user']].groupby(nodeType).count()
        else:
            df = df[[nodeType, content_field]].groupby(nodeType).count()

        df.columns = ['value']
        df = df.reset_index()

        values = df['value'].values.astype(float)

        if np.amin(values) < 0:
            values -= np.amin(values)

        values += 1e-9

        values = np.sort(np.array(values))

        index = np.arange(1,values.shape[0]+1)
        n = values.shape[0]
        g = ((np.sum((2 * index - n  - 1) * values)) / (n * np.sum(values)))

        return g


    def getPalmaCoef(self,nodeType='root', eventTypes=None, content_field="root"):
        """
        Wrapper function calculate the Palma coefficient for the data frame.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - Type of node to calculate the Palma coefficient over.  Options: user or repo (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: Palma coefficient
        """
        result = self.getPalmaCoefHelper(self.main_df, nodeType,eventTypes,content_field)

        return result

    @check_empty(default=None)
    def getPalmaCoefHelper(self, df, nodeType='root', eventTypes=None, content_field = "root"):
        """
        This method returns the Palma coefficient.
        Question #6,14,26
        Input: df - Data frame containing data can be any subset of data
               nodeType - (Optional) This is the node type on whose event counts the Palma coefficient
                          is calculated. Options: user or content (case sensitive)
               eventTypes - A list of event types to include in the calculation
        Output: p - Palma Coefficient
        """
        if eventTypes is not None:
            df = df[df.event.isin(eventTypes)]

        if nodeType != 'user':
            df = df[[nodeType, 'user']].groupby(nodeType).count()
        else:
            df = df[[nodeType, content_field]].groupby(nodeType).count()

        df.columns = ['value']
        df = df.reset_index()

        values = df['value'].values
        values = np.sort(np.array(values))
        percent_nodes = np.arange(1, len(values) + 1) / float(len(values))

        #percent of events taken by top 10% of nodes
        p10 = np.sum(values[percent_nodes >= 0.9])
        #percent of events taken by bottom 40% of nodes
        p40 = np.sum(values[percent_nodes <= 0.4])

        try:
            p = float(p10) / float(p40)
        except ZeroDivisionError:
            return None

        return p


    def getTopKContent(self,content_field='root',k=100,eventTypes=None):
        """
        This method returns the top-k pieces of content by event count for selected event types
        Question #12,13
        Inputs: eventTypes - A list of event types to include in the calculation
                content_field - Options: root, parent, or content.
                k - Number of entities to return
        Outputs: Dataframe with the top-k content ids and their event counts. Columns are content id and the count of that event.
        """
        df = self.main_df.copy()

        if not eventTypes is None:
            df = df[df.event.isin(eventTypes)]
        p = df[[content_field, 'event']].groupby([content_field]).count()
        p = p.sort_values(by='event',ascending=False)
        p.columns = ['value']

        return p.head(k)


    def getDistributionOfEventsByContent(self, content_field='root', eventTypes=['WatchEvent']):
        """
        This method returns the distribution of event type per content e.g. x repos/posts/tweets with y number of events,
        z repos/posts/ with n amounts of events.
        Question #11,12,13
        Inputs: eventTypes - List of event type(s) to get distribution over
        Outputs: Dataframe with the distribution of event type per repo. Columns are repo id and the count of that event.
        """
        df = self.main_df.copy()

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        p = df[[content_field,'time']].groupby(content_field).count()
        p = p.sort_values(by='time')
        p.columns = ['value']
        p = p.reset_index()
        return p


    def getRepoPullRequestAcceptance(self,eventTypes=['PullRequestEvent'],thresh=2):
        """
        Calculate the proportion of pull requests that are accepted for each repo.
        Question #15 (Optional Measurement)
        Inputs: eventTypes: List of event types to include in the calculation (Should be PullRequestEvent).
                thresh: Minimum number of PullRequests a repo must have to be included in the distribution.
        Output: Data frame with the proportion of accepted pull requests for each repo
        """
        #check if optional columns exist
        if not self.main_df_opt is None and 'PullRequestEvent' in self.main_df.event.values:

            df = self.main_df_opt.copy()

            idx = (self.main_df.event.isin(eventTypes)) & (df.merged.isin([True,False,"True","False"]))

            df = df[idx]
            users_repos = self.main_df[idx]

            df['merged'] = df['merged'].map({"True":True,"False":False})

            if len(df) == 0:
                return None

            #subset to only pull requests which are being closed (not opened)
            idx = df['action'] == 'closed'
            closes = df[idx]
            users_repos = users_repos[idx]

            #merge optional columns (action, merged) with the main data frame columns
            closes = pd.concat([users_repos,closes],axis=1)
            closes = closes[['content','merged']]
            closes['value'] = 1

            #create count of accepted (merged) and rejected pull requests by repo
            outcomes = closes.pivot_table(index=['content'],values=['value'],columns=['merged'],aggfunc='sum').fillna(0)

            outcomes.columns = outcomes.columns.get_level_values(1)

            outcomes = outcomes.rename(index=str, columns={True: "accepted", False: "rejected"})

            #if only accepted or reject observed in data, create other column and fill with zero
            for col in ['accepted','rejected']:
                if col not in outcomes.columns:
                    outcomes[col] = 0

            #get total number of pull requests per repo by summing accepted and rejected
            outcomes['total'] = outcomes['accepted'] + outcomes['rejected']
            #get proportion
            outcomes['value'] = outcomes['accepted'] / outcomes['total']

            #subset on content which have enough data
            outcomes = outcomes[outcomes['total'] >= thresh]

            if len(outcomes.index) > 0:
                measurement = outcomes.reset_index()[['content','value']]
            else:
                measurement = None
        else:
            measurement = None

        return measurement


    def getEventTypeRatioTimeline(self, eventTypes=None, event1='IssuesEvent', event2='PushEvent', content_field="root"):
        if self.platform!='reddit':
            df = self.selectedContent.copy()
        else:
            df = self.main_df.copy()

        if eventTypes != None:
            df = df[df['event'].isin(eventTypes)]

        df['value'] = 1

        if len(df.index) < 1:
            return {}

        grouped = df.groupby([content_field, 'user'])

        if len(grouped) > 1:
            measurement = grouped.apply(lambda x: x.value.cumsum()).reset_index()
            measurement['event'] = df['event'].reset_index(drop=True)
        else:
            measurement = df.copy()
            measurement['value'] = df['value'].cumsum()
            measurement['event'] = df['event']

        measurement = measurement[measurement['event'].isin([event1,event2])]

        measurement[event1] = measurement['event'] == event1
        measurement[event2] = measurement['event'] == event2

        measurement['next_event_' + event1] = measurement[event1].shift(-1)
        measurement['next_event_' + event2 ] = measurement[event2].shift(-1)

        bins = np.logspace(-1,3.0,16)
        measurement['num_events_binned'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)

        def ratio(grp):
            if float(grp['next_event_' + event2].sum()) > 0:
                return float(grp['next_event_' + event1].sum()) / float(grp['next_event_' + event2].sum())
            else:
                return 0.0

        if len(measurement.index) > 0:
            measurement = measurement.groupby([content_field,'num_events_binned']).apply(ratio).reset_index()
            measurement.columns = ['content','num_events_binned','value']
        else:
            measurement = None

        measurement = self.getNodeDictionary(measurement)

        return measurement


    def propUserContinue(self,eventTypes=None,content_field="root"):
        if self.platform != 'reddit':
            df = self.selectedContent.copy()
        else:
            df = self.main_df.copy()

        if not eventTypes is None:
            data = df[df['event'].isin(eventTypes)]

        if len(data.index) > 1:
            data['value'] = 1
            grouped = data.groupby(['user',content_field])

            #get running count of user actions on each piece of content
            if grouped.ngroups > 1:
                measurement = grouped.apply(lambda grp: grp.value.cumsum()).reset_index()
            else:
                data['value'] = data['value'].cumsum()
                measurement = data.copy()

            #get total number of user actions on each piece of content
            grouped = measurement.groupby(['user',content_field]).value.max().reset_index()
            grouped.columns = ['user',content_field,'num_events']

            measurement = measurement.merge(grouped,on=['user',content_field])

            #boolean indicator of whether a given event is the last one by the user
            measurement['last_event'] = measurement['value'] == measurement['num_events']

            #bin by the number of previous events
            bins = np.logspace(-1,2.5,30)
            measurement['num_actions'] = pd.cut(measurement['value'],bins).apply(lambda x: np.floor(x.right)).astype(float)
            measurement['last_event']  = ~measurement['last_event']

            #get percentage of events within bin that are NOT the last event for a user
            measurement = measurement.groupby([content_field,'num_actions']).last_event.mean().reset_index()
            measurement.columns = ['content','num_actions','value']
            measurement = self.getNodeDictionary(measurement)

        else:
            measurement = {}

        return measurement


    def determineDf(self, users, eventTypes):
        """
        This function selects a subset of the full data set for a selected set of users and event types.
        Inputs: users - A boolean or a list of users.  If it is list of user ids (login_h) the data frame is subset on only this list of users.
                        If it is True, then the pre-selected node-level subset is used.  If False, then all users are included.
                eventTypes - A list of event types to include in the data set

        Output: A data frame with only the selected users and event types.
        """
        if users==True:
            df = self.selectedUsers
        elif type(users) is list:
            df = df[df.user.isin(users)]
        else:
            df = self.main_df

        if eventTypes!=None:
            df = df[df.event.isin(eventTypes)]

        return df


    def getUserUniqueContent(self, selectedUsers=False, eventTypes=None, content_field="root"):
        """
        This method returns the number of unique repos that a particular set of users contributed too
        Question #17
        Inputs: selectedUsers - A list of users of interest or a boolean indicating whether to subset to the node-level measurement users.
                eventTypes - A list of event types to include in the data
                content_field - CSV column which contains the content ID (e.g. nodeID, parentID, or rootID)
        Output: A dataframe with the user id and the number of repos contributed to
        """
        df = self.determineDf(selectedUsers, eventTypes)
        df = df.groupby('user')
        data = df[content_field].nunique().reset_index()
        data.columns = ['user','value']
        return data


    def getUserActivityTimeline(self, selectedUsers=True,time_bin='1d',cumSum=False,eventTypes=None):
        """
        This method returns the timeline of activity of the desired user over time, either in raw or cumulative counts.
        Question #19
        Inputs: selectedUsers - A list of users of interest or a boolean indicating whether to subset to node-level measurement users.
                time_bin - Time frequency for calculating event counts
                cumSum - Boolean indicating whether to calculate the cumulative activity counts
                eventTypes = List of event types to include in the data
        Output: A dictionary with a data frame for each user with two columns: data and event counts
        """
        df = self.determineDf(selectedUsers,eventTypes)

        df['value'] = 1
        if cumSum:
            df['cumsum'] = df.groupby('user').value.transform(pd.Series.cumsum)
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).max().reset_index()
            df['value'] = df['cumsum']
            df = df.drop('cumsum',axis=1)
        else:
            df = df.groupby(['user',pd.Grouper(key='time',freq=time_bin)]).sum().reset_index()

        data = df.sort_values(['user', 'time'])

        measurements = {}
        for user in data['user'].unique():
            user_df = data[data['user'] == user]
            idx = pd.date_range(min(user_df.time), max(user_df.time))
            user_df = user_df.set_index('time')
            user_df = user_df.reindex(idx)
            user_df.index.names = ['time']
            user_df['user'].ffill(inplace=True)
            user_df['value'].fillna(0,inplace=True)

            measurements[user] = user_df.reset_index()
            del measurements[user]['user']

        return measurements


    def getUserPopularity(self, k=5000, use_metadata=False, eventTypes=None, content_field='root'):
        """
        This method returns the top k most popular users for the dataset, where popularity is measured
        as the total popularity of the repos created by the user.
        Question #25
        Inputs: k - (Optional) The number of users that you would like returned.
                use_metadata - External metadata file containing repo owners.  Otherwise use first observed user with
                               a creation event as a proxy for the repo owner.
                eventTypes - A list of event types to include
        Output: A dataframe with the user ids and number events for that user
        """
        df = self.determineDf(False,eventTypes)

        df['value'] = 1

        content_popularity = df.groupby(content_field)['value'].sum().reset_index()

        creation_event = ''
        if 'CreateEvent' in df.event.unique():
            creation_event = 'CreateEvent'
        elif 'post' in df.event.unique():
            creation_event = 'post'
        elif 'tweet' in df.event.unique():
            creation_event = 'tweet'

        if use_metadata:
            #merge content popularity with the owner information in content_metadata
            #drop data for which no owner information exists in metadata
            merged = content_popularity.merge(self.repoMetaData,left_on=content_field,right_on='full_name_h',
                                           how='left').dropna()
        elif df[content_field].str.match('.{22}/.{22}').all():
            #if all content IDs have the correct format use the owner info from the content id
            content_popularity['owner_id'] = content_popularity[content_field].apply(lambda x: x.split('/')[0])
        elif creation_event != '':
            #otherwise use creation event as a proxy for ownership
            user_content = df[df['event'] == creation_event].sort_values('time').drop_duplicates(subset=content_field,keep='first')
            user_content = user_content[['user',content_field]]
            user_content.columns = ['owner_id', content_field]
            if len(user_content.index) >= 0:
                content_popularity = user_content.merge(content_popularity,on=content_field,how='left')
            else:
                return None
        else:
            return None

        measurement = content_popularity.groupby('owner_id').value.sum().sort_values(ascending=False).head(k)
        measurement = pd.DataFrame(measurement).sort_values('value',ascending=False).reset_index()

        return measurement


    def getAvgTimebwEventsUsers(self, selectedUsers=True, nCPU=1):
        """
        This method returns the average time between events for each user

        Inputs: df - Data frame of all data for repos
        users - (Optional) List of specific users to calculate the metric for
        nCPu - (Optional) Number of CPU's to run metric in parallel
        Outputs: A list of average times for each user. Length should match number of repos
        """
        df = self.determineDf(selectedUsers)
        users = self.df['user'].unique()
        args = [(df, users[i]) for i, item_a in enumerate(users)]
        pool = pp.ProcessPool(nCPU)
        deltas = pool.map(self.getMeanTimeHelper, args)

        return deltas


    def getMeanTimeUser(self,df, user):
        """
        Helper function for getting the average time between events

        Inputs: Same as average time between events
        Output: Same as average time between events
        """
        d = df[df.user == user]
        d = d.sort_values(by='time')
        delta = np.mean(np.diff(d.time)) / np.timedelta64(1, 's')
        return delta


    def getMeanTimeUserHelper(self, args):

        return self.getMeanTimeUser(*args)


    def getUserDiffusionDelay(self,unit='h', selectedUser=True,eventTypes=None):
        """
        This method returns distribution the diffusion delay for each user
        Question #27
        Inputs: DataFrame - Desired dataset
        unit - (Optional) This is the unit that you want the distribution in. Check np.timedelta64 documentation
        for the possible options
        metadata_file - File containing user account creation times.  Otherwise use first observed action of user as proxy for account creation time.
        Output: A list (array) of deltas in units specified
        """
        df = self.determineDf(selectedUser,eventTypes)

        df['value'] = df['time']
        df['value'] = pd.to_datetime(df['value'])
        df['value'] = df['value'].dt.round('1H')

        if self.useUserMetaData:
            df = df.merge(self.UserMetaData[['user','created_at']],left_on='user',right_on='user',how='left')
            df = df[['user','created_at','value']].dropna()
            measurement = df['value'].sub(df['created_at']).apply(lambda x: int(x / np.timedelta64(1, unit)))
        else:
            grouped = df.groupby('user')
            transformed = grouped['value'].transform('min')
            measurement = df['value'].sub(transformed).apply(lambda x: int(x / np.timedelta64(1, unit)))
        return measurement


    def getMostActiveUsers(self,k=5000,eventTypes=None):
        """
        This method returns the top k users with the most events.
        Question #24b
        Inputs: DataFrame - Desired dataset. Used mainly when dealing with subset of events
        k - Number of users to be returned
        Output: Dataframe with the user ids and number of events
        """
        df = self.main_df

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        df['value'] = 1
        df = df.groupby('user')
        measurement = df.value.sum().sort_values(ascending=False).head(k)
        measurement = pd.DataFrame(measurement).sort_values('value',ascending=False).reset_index()
        return measurement


    def getUserActivityDistribution(self,eventTypes=None,selectedUser=False):
        """
        This method returns the distribution for the users activity (event counts).
        Question #24a
        Inputs: DataFrame - Desired dataset
        eventTypes - (Optional) Desired event type to use
        Output: List containing the event counts per user
        """
        if selectedUser:
            df = self.selectedUsers
        else:
            df = self.main_df

        if eventTypes != None:
            df = df[df.event.isin(eventTypes)]

        df['value'] = 1
        df = df.groupby('user')
        measurement = df.value.sum().reset_index()

        return measurement


    def getUserPullRequestAcceptance(self,eventTypes=['PullRequestEvent'], thresh=2):
        """
        Calculate the proportion of pull requests that are accepted by each user.
        Question #15 (Optional Measurement)
        Inputs: eventTypes: List of event types to include in the calculation (Should be PullRequestEvent).
                thresh: Minimum number of PullRequests a repo must have to be included in the distribution.
        Output: Data frame with the proportion of accepted pull requests for each user
        """
        if not self.main_df_opt is None and 'PullRequestEvent' in self.main_df.event.values:
            df = self.main_df_opt.copy()

            idx = (self.main_df.event.isin(eventTypes)) & (df.merged.isin([True,False,"True","False"]))
            df = df[idx]

            df['merged'] = df['merged'].map({"False":False,"True":True})
            users_repos = self.main_df[idx]

            if len(df) == 0:
                return None

            #subset on only PullRequest close actions (not opens)
            idx = df['action'] == 'closed'
            closes = df[idx]
            users_repos = users_repos[idx]

            #merge pull request columns (action, merged) with main data frame columns
            closes = pd.concat([users_repos,closes],axis=1)
            closes = closes[['user','content','merged']]
            closes['value'] = 1

            #add up number of accepted (merged) and rejected pullrequests by user and repo
            outcomes = closes.pivot_table(index=['user','content'],values=['value'],columns=['merged'],aggfunc=np.sum).fillna(0)

            outcomes.columns = outcomes.columns.get_level_values(1)

            outcomes = outcomes.rename(index=str, columns={True: "accepted", False: "rejected"})

            for col in ['accepted','rejected']:
                if col not in outcomes.columns:
                    outcomes[col] = 0

            outcomes['total'] = outcomes['accepted'] +  outcomes['rejected']
            outcomes['value'] = outcomes['accepted'] / outcomes['total']
            outcomes = outcomes.reset_index()
            outcomes = outcomes[outcomes['total'] >= thresh]

            if len(outcomes.index) > 0:
                #calculate the average acceptance rate for each user across their repos
                measurement = outcomes[['user','value']].groupby('user').mean().reset_index()
            else:
                measurement = None
        else:
            measurement = None

        return measurement
