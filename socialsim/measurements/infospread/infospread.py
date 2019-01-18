import pandas as pd
import pickle as pkl

from .infospread_node       import InfospreadNode
from .infospread_population import InfospreadPopulation
from .infospread_community  import InfospreadCommunity

from ..measurements import MeasurementsBaseClass

class InfospreadMeasurements(MeasurementsBaseClass, InfospreadNode,
    InfospreadPopulation, InfospreadCommunity):
    def __init__(self, dataset, configuration, metadata, content_node_ids=[],
        user_node_ids=[], metaContentData=False, metaUserData=False,
        community_dictionary='', platform='github'):
        """
        Description:

        Input:
            :dataset:
            :configuration:
            :metadata:

        Output:
            None
        """
        MeasurementsBaseClass.__init__(self, dataset, configuration)

        self.platform = platform

        # What are these used for?
        self.contribution_events = ['PullRequestEvent', 'PushEvent',
            'IssuesEvent', 'IssueCommentEvent', 'PullRequestReviewCommentEvent',
            'CommitCommentEvent', 'CreateEvent', 'post', 'tweet']

        # What are these used for?
        self.popularity_events = ['WatchEvent', 'ForkEvent', 'comment', 'post',
            'retweet', 'quote', 'reply']

        self.main_df = self.preprocess(dataset)

        # store action and merged columns in a seperate data frame that is not used for most measurements
        if platform == 'github' and len(self.main_df.columns) == 6 and 'action' in self.main_df.columns:
            self.main_df_opt = self.main_df.copy()[['action', 'merged']]
            self.main_df = self.main_df.drop(['action', 'merged'], axis=1)
        else:
            self.main_df_opt = None

        # For content centric
        if content_node_ids != ['all']:
            if self.platform == 'reddit':
                self.selectedContent = self.main_df[self.main_df.root.isin(content_node_ids)]
            elif self.platform == 'twitter':
                self.selectedContent = self.main_df[self.main_df.root.isin(content_node_ids)]
            else:
                self.selectedContent = self.main_df[self.main_df.content.isin(content_node_ids)]
        else:
            self.selectedContent = self.main_df

        # For userCentric
        self.selectedUsers = self.main_df[self.main_df.user.isin(user_node_ids)]

        if metaContentData!=False:
            self.useContentMetaData = True
            meta_content_data = pd.read_csv(metaContentData)
            self.contentMetaData = self.preprocessContentMeta(meta_content_data)
        else:
            self.useContentMetaData = False
        if metaUserData != False:
            self.useUserMetaData = True
            self.userMetaData = self.preprocessUserMeta(pd.read_csv(metaUserData))
        else:
            self.useUserMetaData = False

        # For community measurements
        self.community_dict_file = community_dictionary
        if self.platform == 'github':
            self.communityDF = self.getCommmunityDF(community_col='community')
        elif self.platform == 'reddit':
            self.communityDF = self.getCommmunityDF(community_col='subreddit')
        else:
            self.communityDF = self.getCommmunityDF(community_col='')

    def preprocess(self, df):
        """
        Description:

        Input:

        Output:
        Edit columns, convert date, sort by date
        """

        if self.platform=='reddit':
            mapping = {'actionType'   : 'event',
                       'communityID'  : 'subreddit',
                       'keywords'     : 'keywords',
                       'nodeID'       : 'content',
                       'nodeTime'     : 'time',
                       'nodeUserID'   : 'user',
                       'parentID'     : 'parent',
                       'rootID'       : 'root'}
        elif self.platform=='twitter':
            mapping = {'actionType'   : 'event',
                       'nodeID'       : 'content',
                       'nodeTime'     : 'time',
                       'nodeUserID'   : 'user',
                       'parentID'     : 'parent',
                       'rootID'       : 'root'}
        elif self.platform=='github':
            mapping = {'nodeID'       : 'content',
                       'nodeUserID'   : 'user',
                       'actionType'   : 'event',
                       'nodeTime'     : 'time',
                       'actionSubType': 'action',
                       'status'       : 'merged'}

        df = df.rename(index=str, columns=mapping)

        df = df[df.event.isin(self.popularity_events + self.contribution_events)]

        df = df.sort_values(by='time')
        df = df.assign(time=df.time.dt.floor('h'))
        return df

    def preprocessContentMeta(self, df):
        try:
            df.columns = ['content', 'created_at', 'owner_id', 'language']
        except:
            df.columns = ['created_at', 'owner_id', 'content']

        df['created_at'] = pd.to_datetime(df['created_at'])

        df = df[df.content.isin(self.main_df.content.values)]

        return df

    def preprocessUserMeta(self, df):
        try:
            df.columns = ['user', 'created_at', 'location', 'company']
        except:
            df.columns = ['user', 'created_at', 'city', 'country', 'company']

        df['created_at'] = pd.to_datetime(df['created_at'])

        df = df[df.user.isin(self.main_df.user.values)]

        return df

    def readPickleFile(self, ipFile):

        with open(ipFile, 'rb') as handle:
            obj = pkl.load(handle)

        return obj
