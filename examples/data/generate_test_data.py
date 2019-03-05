import json
import os 
import sys

import string 
import random

import pandas as pd

class Generator:
    def __init__(self):

        k = 10000

        last30 = pd.datetime.now().replace(microsecond=0) - pd.Timedelta('30H')

        self.informationIDs = list(set([f() for _ in range(k)]))

        self.pool = {
            'nodeID'     : [f() for _ in range(int(k/10))],
            'nodeUserID' : [f() for _ in range(int(k/10))],
            'urlDomains' : [f() for _ in range(int(k/100))],
            'actionType' : [f() for _ in range(20)],
            'nodeTime'   : pd.date_range(last30, periods=30*60*60, freq='S')
        }

        self.platforms = ['github', 'reddit', 'twitter']

        return None 


    def generate_test_data(self, dataset_directory):
        """
        Description: Generates 

        Input:
            :dataset_directory:

        Output:
            None

        """

        test_data = dataset_directory + 'test_data.json'

        count = 0

        for informationID in self.informationIDs:

            print(count)
            count += 1

            platform = random.choice(self.platforms)

            datapoint = self.make_point(platform, informationID)
            datapoint = json.dumps(datapoint) + '\n'

            with open(test_data, '+a') as f:
                f.write(datapoint)

        return None 


    def make_point(self, platform, informationID):
        if platform=='reddit':
            datapoint = self.make_reddit_point(informationID)
        elif platform=='twitter':
            datapoint = self.make_twitter_point(informationID)
        elif platform=='github':
            datapoint = self.make_github_point(informationID)

        return datapoint 


    def make_reddit_point(self, informationID):
        platform ='reddit'

        nodeUserID = random.choice(self.pool['nodeUserID'])

        nodeID   = random.choice(self.pool['nodeID'])
        parentID = random.choice(self.pool['nodeID'])
        rootID   = random.choice(self.pool['nodeID'])

        actionType = random.choice(self.pool['actionType'])

        urlDomains = random.choice([0, 1, 2])

        if urlDomains==0:
            urlDomains=[]
        else:
            urlDomains = random.choices(self.pool['urlDomains'], k=urlDomains)

        nodeTime = str(random.choice(self.pool['nodeTime']))

        datapoint = {
            'nodeUserID'    : nodeUserID,
            'informationID' : informationID,
            'nodeID'        : nodeID,
            'nodeTime'      : nodeTime,
            'actionType'    : actionType,
            'platform'      : platform,
            'parentID'      : parentID,
            'rootID'        : rootID,
            'urlDomains'    : urlDomains
        }

        return datapoint 


    def make_twitter_point(self, informationID):
        platform ='twitter'

        nodeUserID = random.choice(self.pool['nodeUserID'])
        nodeID     = random.choice(self.pool['nodeID'])
        parentID   = random.choice(self.pool['nodeID'])
        rootID     = random.choice(self.pool['nodeID'])
        actionType = random.choice(self.pool['actionType'])

        urlDomains = random.choice([0, 1, 2])

        if urlDomains==0:
            urlDomains=[]
        else:
            urlDomains = random.choices(self.pool['urlDomains'], k=urlDomains)

        nodeTime = str(random.choice(self.pool['nodeTime']))

        datapoint = {
            'urlDomains'    : urlDomains,
            'nodeUserID'    : nodeUserID,
            'informationID' : informationID,
            'nodeID'        : nodeID,
            'nodeTime'      : nodeTime,
            'actionType'    : actionType,
            'platform'      : platform,
            'parentID'      : parentID,
            'rootID'        : rootID
        }

        return datapoint 


    def make_github_point(self, informationID):
        platform ='github'

        nodeUserID = random.choice(self.pool['nodeUserID'])
        nodeID     = random.choice(self.pool['nodeID'])
        actionType = random.choice(self.pool['actionType'])
        urlDomains = random.choice([0, 1, 2])

        if urlDomains==0:
            urlDomains=[]
        else:
            urlDomains = random.choices(self.pool['urlDomains'], k=urlDomains)

        nodeTime = str(random.choice(self.pool['nodeTime']))

        datapoint = {
            'urlDomains'    : urlDomains,
            'nodeUserID'    : nodeUserID,
            'informationID' : informationID,
            'nodeID'        : nodeID,
            'nodeTime'      : nodeTime,
            'actionType'    : actionType,
            'platform'      : platform
        }

        return datapoint 


def f(n=10):
    a = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
    return a


if __name__=='__main__':
    generator = Generator()

    generator.generate_test_data('./')