import pandas as pd

import pprint
import re
import json
import itertools
#import networkx as nx
import numpy as np

from datetime import datetime

from .twitter_cascade_reconstruction import full_reconstruction,get_reply_cascade_root_tweet


def load_json(fn):

    json_data = []
    
    if type(fn) == str:
        with open(fn,'rb') as f:
            for line in f:
                json_data.append(json.loads(line))
    else:
        for fn0 in fn:
            with open(fn0,'rb') as f:
                for line in f:
                    json_data.append(json.loads(line))

    return(json_data)
                
def convert_timestamps(dataset,timestamp_field = "nodeTime"):

    """
    Converts all timestamps to ISO 8601 formatted strings
    """
    
    try:
        dataset[timestamp_field] = pd.to_datetime(dataset[timestamp_field], unit='s')
    except:
        try:
            dataset[timestamp_field] = pd.to_datetime(dataset[timestamp_field], unit='ms')
        except:
            dataset[timestamp_field] = pd.to_datetime(dataset[timestamp_field])

    dataset[timestamp_field] = dataset[timestamp_field].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            
    return(dataset)

def get_info_id_from_text(text_list = [], keywords = []):

    word_list = r"\b" + keywords[0] + r"\b"
    for w in keywords[1:]:
        word_list += "|" + r"\b" + w + r"\b"

    info_ids = []
    for text in text_list:
        info_ids += re.findall(word_list,text)
        
    return(list(set(info_ids)))
        
def get_info_id_from_fields(row, fields=['entities.hashtags.text']):

    """
    Extract information IDs from specified fields in the JSON

    :param row: A DataFrame row containing the JSON fields
    :param fields: A list of field paths from which to extract the info IDs, e.g. entities.hashtags.text, entities.user_mentions.screen_name
    :returns: a list of information IDs that are in the specified fields
    """
    
    info_ids = []
    for path in fields:
        path = path.split('.')

        val = row.copy()

        for i,f in enumerate(path):
            
            if (isinstance(val,pd.Series) or type(val) == dict) and f in val.keys():
                #move down JSON path
                val = val[f]

            if type(val) == list:
                #iterate over list
                for v in val:
                    if type(v) == dict:
                        v = v[path[i+1]]
                    info_ids.append(v)
                break
            elif i == len(path) and type(val) == str:
                info_ids.append(val)
                
    return list(set(info_ids))


def extract_telegram_data(fn='telegram_data.json',
                          info_id_fields=None,
                          keywords = [],
                          anonymized=False):

    """
    Extracts fields from Telegram JSON data

    :param fn: A filename or list of filenames which contain the JSON Telegram data
    :param info_id_fields: A list of field paths from which to extract the information IDs. If None, don't extract any.
    """
    
    json_data = load_json(fn)
    data = pd.DataFrame(json_data)

    get_info_ids = False
    if not info_id_fields is None or len(keywords) > 0:
        get_info_ids = True
    
    if anonymized:
        name_suffix = "_h"
        text_suffix = "_m"
    else:
        name_suffix = ""
        text_sufix = ""
        
    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime',
                      'platform','communityID']
    if get_info_ids:
        output_columns.append('informationIDs')

    
    print('Extracting fields...')

    if len(keywords) > 0:
        data.loc[:,'informationIDs'] = data['doc'].apply(lambda x: get_info_id_from_text([x['text' + text_suffix]], keywords))
    elif not info_id_fields is None:
        data.loc[:,'informationIDs'] = pd.Series([get_info_id_from_fields(c,info_id_fields,dict_field=True) for i,c in data.iterrows()])

    data = data.drop_duplicates('uid' + name_suffix)
    
    data.loc[:,'actionType']=['message']*len(data)

    data.loc[:,'nodeTime'] = data['norm'].apply(lambda x: x['timestamp'])
    
    data.loc[:,'communityID'] = data['doc'].apply(lambda x: x['peer']['username'] if 'peer' in x.keys() else None)

    data.loc[:,'nodeID'] = data['doc'].apply(lambda x: str(x['to_id']['channel_id']) + '_' + str(x['id']))

    data.loc[:,'nodeUserID'] = data['doc'].apply(lambda x: x['from_id' + name_suffix] if 'from_id' + name_suffix in x.keys() else None)
    data.loc[data['nodeUserID'].isnull(),'nodeUserID'] = data.loc[data['nodeUserID'].isnull(),'norm'].apply(lambda x: x['author'])
    
    data.loc[:,'platform'] = 'telegram'
    
    data.loc[:,'parentID'] = data['doc'].apply(lambda x: str(x['fwd_from']['channel_id']) + '_' + str(x['fwd_from']['channel_post']) if 'fwd_from' in x.keys() and not x['fwd_from'] is None and not x['fwd_from']['channel_id'] is None and not x['fwd_from']['channel_post'] is None else None)

    data.loc[:,'parentID'] = data['doc'].apply(lambda x: str(x['to_id']['channel_id']) + '_' + str(x['reply_to_msg_id']) if 'reply_to_msg_id' in x.keys() and not x['reply_to_msg_id'] is None else None)

    data.loc[:,'rootID'] = '?'
    data.loc[data['parentID'].isna(),'rootID'] = data.loc[data['parentID'].isna(),'nodeID']

    data.loc[data['parentID'].isna(),'parentID'] = data.loc[data['parentID'].isna(),'nodeID']

    data = data[data['parentID'].isin(list(set(data['nodeID'])))]

    data = data[output_columns]
    
    data = get_reply_cascade_root_tweet(data)
        
    #remove broken portions
    data = data[data['rootID'].isin(list(set(data['nodeID'])))]

    print('Sorting...')
    data = data.sort_values('nodeTime').reset_index(drop=True)            

    #initialize info ID column with empty lists
    data['threadInfoIDs'] = [[] for i in range(len(data))]
    
    #for some reason having a non-object column in the dataframe messes up the assignment of lists to individual cell values
    #remove it temporarily and add back later
    nodeTimes = data['nodeTime']
    data = data[[c for c in data.columns if c != 'nodeTime']]
    
    #get children of node
    def get_children(nodeID):

        children = data[data['parentID'] == nodeID]['nodeID']
        children = children[children.values != nodeID]
        
        return(children)


    #all replies/fwds of a message mentioning a unit of information are also assigned that unit of information
    def add_info_to_children(nodeID,list_info=[]):

        infos = list(data[data['nodeID'] == nodeID]['informationIDs'].values[0])

        list_info = list_info.copy()

        children = get_children(nodeID)
        
        if len(children) > 0:

            list_info += infos
    
            if len(list_info) > 0 and len(children) > 1:
                data.loc[children.index.values,'threadInfoIDs'] = [list_info for i in range(len(children))]
            elif len(list_info) > 0 and len(children) == 1:
                data.at[children.index[0],'threadInfoIDs'] = list_info

            for child in children.values:
                add_info_to_children(child,list_info)

    if get_info_ids:
        print('Adding information IDs to children...')
        #for each thread in data, propagate infromation IDs to children
        roots = data['rootID'].unique()
        for r,root in enumerate(roots):
            add_info_to_children(root)
            if r % 50 == 0:
                print('{}/{}'.format(r,len(roots)))

    data['nodeTime'] = nodeTimes

    if get_info_ids:
        data['informationIDs'] = data.apply(lambda x: list(set(x['informationIDs'] + x['threadInfoIDs'])),axis=1)
    
        data = data[data['informationIDs'].str.len() > 0]
        
        print('Expanding events...')
        #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
        s = data.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'informationID'

        data = data.drop('informationIDs', axis=1).join(s).reset_index(drop=True)

    data = data.drop('threadInfoIDs',axis=1)
    data = data.sort_values('nodeTime').reset_index(drop=True)
    data = convert_timestamps(data)
    data = data[~data['communityID'].isnull()]
    
    print('Done!')
    return data
   

def extract_reddit_data(fn='reddit_data.json',
                        info_id_fields=None,
                        keywords = [],
                        anonymized=False):

    """
    Extracts fields from Reddit JSON data

    :param fn: A filename or list of filenames which contain the JSON Reddit data
    :param info_id_fields: A list of field paths from which to extract the information IDs. If None, don't extract any.
    """

    json_data = load_json(fn)
    data = pd.DataFrame(json_data)

    get_info_ids = False
    if not info_id_fields is None or len(keywords) > 0:
        get_info_ids = True
    
    if anonymized:
        name_suffix = "_h"
        text_suffix = "_m"
    else:
        name_suffix = ""
        text_suffix = ""
    
    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType',
                      'nodeTime','platform','communityID']
    if get_info_ids:
        output_columns.append('informationIDs')
                                  
    print('Extracting fields...')
    if len(keywords) > 0:
        data['text'] = data['body' + text_suffix].replace(np.nan, '', regex=True) + data['selftext' + text_suffix].replace(np.nan, '', regex=True) + data['title' + text_suffix].replace(np.nan, '', regex=True)
        data.loc[:,'informationIDs'] = data['text'].apply(lambda x: get_info_id_from_text([x], keywords))
    elif not info_id_fields is None:
        data.loc[:,'informationIDs'] = pd.Series([get_info_id_from_fields(c,info_id_fields) for i,c in data.iterrows()])
        data['n_info_ids'] = data['informationIDs'].apply(len)
        data = data.sort_values("n_info_ids",ascending=False)

    data = data.drop_duplicates('id' + name_suffix)
    
    data.rename(columns={'id' + name_suffix:'nodeID','author' + name_suffix:'nodeUserID',
                         'created_utc':'nodeTime','parent_id' + name_suffix:'parentID','link_id' + name_suffix:'rootID'}, inplace=True)
    
    data.loc[:,'actionType']=['comment']*len(data)
    data.loc[~data["title_m"].isnull(),'actionType'] = 'post'
    
    data.loc[data['actionType'] == "comment",'nodeID']=['t1_' + x for x in data.loc[data['actionType'] == "comment",'nodeID']]
    data.loc[data['actionType'] == "post",'nodeID']=['t3_' + x for x in data.loc[data['actionType'] == "post",'nodeID']]

    data.loc[data['actionType'] == "post",'rootID'] = data.loc[data['actionType'] == "post",'nodeID']
    data.loc[data['actionType'] == "post",'parentID'] = data.loc[data['actionType'] == "post",'nodeID']

    data.loc[:,'communityID'] = data['subreddit_id']

    data.loc[:,'platform'] = 'reddit'
    
    #remove broken portions
    data = data[data['parentID'].isin(list(set(data['nodeID'])))]
    data = data[data['rootID'].isin(list(set(data['nodeID'])))]

    print('Sorting...')
    data = data.sort_values('nodeTime').reset_index(drop=True)            

    data = data[output_columns]
    
    #initialize info ID column with empty lists
    data['threadInfoIDs'] = [[] for i in range(len(data))]
    
    #for some reason having a non-object column in the dataframe messes up the assignment of lists to individual cell values
    #remove it temporarily and add back later
    nodeTimes = data['nodeTime']
    data = data[[c for c in data.columns if c != 'nodeTime']]
    
    
    #get children of node
    def get_children(nodeID):

        children = data[data['parentID'] == nodeID]['nodeID']
        children = children[children.values != nodeID]
        
        return(children)

    print(data)
    # all comments on a post/comment mentioning a unit of information are also assigned that unit of information
    def add_info_to_children(nodeID,list_info=[]):

        infos = list(data[data['nodeID'] == nodeID]['informationIDs'].values[0])

        list_info = list_info.copy()

        children = get_children(nodeID)
        
        if len(children) > 0:

            list_info += infos
    
            if len(list_info) > 0 and len(children) > 1:
                data.loc[children.index.values,'threadInfoIDs'] = [list_info for i in range(len(children))]
            elif len(list_info) > 0 and len(children) == 1:
                data.at[children.index[0],'threadInfoIDs'] = list_info

            for child in children.values:
                add_info_to_children(child,list_info)

    if get_info_ids:
        print('Adding information IDs to children...')
        #for each thread in data, propagate infromation IDs to children
        roots = data['rootID'].unique()
        for r,root in enumerate(roots):
            add_info_to_children(root)
            if r % 50 == 0:
                print('{}/{}'.format(r,len(roots)))

            
    data['nodeTime'] = nodeTimes

    if get_info_ids:
        data['informationIDs'] = data.apply(lambda x: list(set(x['informationIDs'] + x['threadInfoIDs'])),axis=1)
    
        data = data[data['informationIDs'].str.len() > 0]
    
        print('Expanding events...')
        #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
        s = data.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'informationID'
    
        data = data.drop('informationIDs', axis=1).join(s).reset_index(drop=True)

    data = data.drop('threadInfoIDs',axis=1)
    data = data.sort_values('nodeTime').reset_index(drop=True)
    data = convert_timestamps(data)

    print('Done!')
    return data
    

def extract_twitter_data(fn='twitter_data.json',
                         info_id_fields=None,
                         keywords = [],
                         anonymized=False):

    """
    Extracts fields from Twitter JSON data

    :param fn: A filename or list of filenames which contain the JSON Twitter data
    :param info_id_fields: A list of field paths from which to extract the information IDs. If None, don't extract any.
    :param keywords:
    :params anonymized: Whether the data is in raw Twitter API format (False) or if it is in the processed and anonymized SocialSim data format (True).  The anonymized format has several modifications to field names.
    """
    
    json_data = load_json(fn)
    data = pd.DataFrame(json_data)

    get_info_ids = False
    if not info_id_fields is None or len(keywords) > 0:
        get_info_ids = True

    
    if anonymized:
        name_suffix = "_h"
        text_suffix = "_m"
    else:
        name_suffix = ""
        text_suffix = ""
    
    data = data.sort_values("timestamp_ms").reset_index(drop=True)

    output_columns = ['nodeID', 'nodeUserID', 'parentID', 'rootID', 'actionType', 'nodeTime',
                      'partialParentID','platform']
    if get_info_ids:
        output_columns.append('informationIDs')
    
    print('Extracting fields...')
    tweets = data
    if len(keywords) > 0:
        data.loc[:,'informationIDs'] = data['text' + text_suffix].apply(lambda x: get_info_id_from_text([x], keywords))
    elif not info_id_fields is None:
        tweets.loc[:,'informationIDs'] = pd.Series([get_info_id_from_fields(t,info_id_fields) for i,t in tweets.iterrows()])
        tweets.loc[:,'n_info_ids'] = tweets['informationIDs'].apply(len)
        tweets = tweets.sort_values('n_info_ids',ascending=False).reset_index(drop=True)

    tweets = tweets.drop_duplicates('id_str' + name_suffix)
    
    tweets.rename(columns={'id_str' + name_suffix: 'nodeID',
                           'timestamp_ms': 'nodeTime'}, inplace=True)


    tweets.loc[:,'platform'] = 'twitter'
    tweets.loc[:,'nodeTime'] = pd.to_datetime(tweets['nodeTime'],unit='ms')
    tweets.loc[:,'nodeTime'] = tweets['nodeTime'].apply(lambda x: datetime.strftime(x,'%Y-%m-%dT%H:%M:%SZ'))

    tweets.loc[:,'nodeUserID'] = tweets['user'].apply(lambda x: x['id_str' + name_suffix])
    
    tweets.loc[:,'is_reply'] = (tweets['in_reply_to_status_id_str' + name_suffix] != '') & (~tweets['in_reply_to_status_id_str' + name_suffix].isna())

    if 'retweeted_status.in_reply_to_status_id_str' + name_suffix not in tweets:
        tweets.loc[:,'retweeted_status.in_reply_to_status_id_str' + name_suffix] = ''
    if 'quoted_status.in_reply_to_status_id_str' + name_suffix not in tweets:
        tweets.loc[:,'quoted_status.in_reply_to_status_id_str' + name_suffix] = ''
    if 'quoted_status.is_quote_status' not in tweets:
        tweets.loc[:,'quoted_status.is_quote_status'] = False
    if 'quoted_status' not in tweets:
        tweets.loc[:,'quoted_status'] = None
        
    #keep track of specific types of reply chains (e.g. retweet of reply, retweet of quote of reply) because the parents and roots will be assigned differently
    tweets.loc[:,'is_retweet_of_reply'] = (~tweets['retweeted_status.in_reply_to_status_id_str' + name_suffix].isna()) & (~(tweets['retweeted_status.in_reply_to_status_id_str' + name_suffix] == ''))
    tweets.loc[:,'is_retweet_of_quote'] = (~tweets['retweeted_status'].isna()) & (~tweets['quoted_status'].isna()) & (tweets['quoted_status.in_reply_to_status_id_str' + name_suffix] == '')              
    tweets.loc[:,'is_retweet_of_quote_of_reply'] = (~tweets['retweeted_status'].isna()) & (~tweets['quoted_status'].isna()) & (~(tweets['quoted_status.in_reply_to_status_id_str' + name_suffix] == ''))
    tweets.loc[:,'is_retweet'] = (~tweets['retweeted_status'].isna()) & (~tweets['is_retweet_of_reply']) & (~tweets['is_retweet_of_quote']) & (~tweets['is_retweet_of_quote_of_reply'])

    
    tweets.loc[:,'is_quote_of_reply'] = (~tweets['quoted_status.in_reply_to_status_id_str' + name_suffix].isna()) & (~(tweets['quoted_status.in_reply_to_status_id_str' + name_suffix] == '')) & (tweets['retweeted_status'].isna())
    tweets.loc[:,'is_quote_of_quote'] = (~tweets['quoted_status.is_quote_status'].isna()) & (tweets['quoted_status.is_quote_status'] == True) & (tweets['retweeted_status'].isna())
    tweets.loc[:,'is_quote'] = (~tweets['quoted_status'].isna()) & (~tweets['is_quote_of_reply']) & (~tweets['is_quote_of_quote']) & (tweets['retweeted_status'].isna()) & (~tweets['is_reply']) 

    tweets.loc[:,'is_orig'] = (~tweets['is_reply']) & (~tweets['is_retweet']) & (~tweets['is_quote']) & (~tweets['is_quote_of_reply']) & (~tweets['is_quote_of_quote']) & (~tweets['is_retweet_of_reply']) & (~tweets['is_retweet_of_quote_of_reply']) & (~tweets['is_retweet_of_quote'])

    
    tweet_types = ['is_reply','is_retweet','is_quote','is_orig','is_retweet_of_reply','is_retweet_of_quote','is_retweet_of_quote_of_reply','is_quote_of_reply','is_quote_of_quote']
   
    to_concat = []

    replies = tweets[tweets['is_reply']]
    if len(replies) > 0:
        #for replies we know immediate parent but not root
        replies.loc[:,'actionType'] = 'reply'
        replies.loc[:,'parentID'] = tweets['in_reply_to_status_id_str' + name_suffix]
        replies.loc[:,'rootID'] = '?'
        replies.loc[:,'partialParentID'] = tweets['in_reply_to_status_id_str' + name_suffix]

        to_concat.append(replies)

    retweets = tweets[ (tweets['is_retweet']) & (~tweets['is_quote']) ]
    if len(retweets) > 0:
        #for retweets we know the root but not the immediate parent
        retweets.loc[:,'actionType'] = 'retweet'
        retweets.loc[:,'rootID'] = retweets['retweeted_status'].apply(lambda x: x['id_str' + name_suffix])
        retweets.loc[:,'parentID'] = '?'
        retweets.loc[:,'partialParentID'] = retweets['retweeted_status'].apply(lambda x: x['id_str' + name_suffix])

        to_concat.append(retweets)
        
    retweets_of_replies = tweets[ tweets['is_retweet_of_reply'] ]
    if len(retweets_of_replies) > 0:
        #for retweets of replies the "root" is actually the reply not the ultimate root
        #the parent of a retweet of a reply will be the reply or any retweet of the reply
        #the root can be retraced by following parents up the tree
        retweets_of_replies.loc[:,'parentID'] = '?'
        retweets_of_replies.loc[:,'rootID'] = '?'
        retweets_of_replies.loc[:,'partialParentID'] = retweets_of_replies['retweeted_status'].apply(lambda x: x['in_reply_to_status_id_str' + name_suffix])
        retweets_of_replies.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_replies)

    retweets_of_quotes = tweets[ tweets['is_retweet_of_quote'] ]
    if len(retweets_of_quotes) > 0:
        #for retweets of quotes we know the root (from the quoted status) but not the parent
        #the parent will be either the quote or any retweets of it
        retweets_of_quotes.loc[:,'parentID'] = '?'
        retweets_of_quotes.loc[:,'rootID'] = retweets_of_quotes['quoted_status'].apply(lambda x: x['id_str' + name_suffix])
        retweets_of_quotes.loc[:,'partialParentID'] = retweets_of_quotes['retweeted_status'].apply(lambda x: x['id_str' + name_suffix])
        retweets_of_quotes.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_quotes)

    retweets_of_quotes_of_replies = tweets[ tweets['is_retweet_of_quote_of_reply'] ]
    if len(retweets_of_quotes_of_replies) > 0:
        #for retweets of quotes of replies we don't know the root or the parent. the quoted status refers back to the reply not the final root
        #the parent will be either the quote or a retweet of the quote
        #we can find the root by tracking parents up the tree
        retweets_of_quotes_of_replies.loc[:,'parentID'] = '?'
        retweets_of_quotes_of_replies.loc[:,'rootID'] = '?'
        retweets_of_quotes_of_replies.loc[:,'partialParentID'] = retweets_of_quotes_of_replies['quoted_status'].apply(lambda x: x['id_str' + name_suffix])
        retweets_of_quotes_of_replies.loc[:,'actionType'] = 'retweet'

        to_concat.append(retweets_of_quotes_of_replies)
                                                                                                                                       
    quotes = tweets[tweets['is_quote']]
    if len(quotes) > 0:
        #for quotes we know the root but not the parent
        quotes.loc[:,'actionType'] = 'quote'
        quotes.loc[:,'rootID'] = quotes['quoted_status'].apply(lambda x: x['id_str' + name_suffix])
        quotes.loc[:,'parentID'] = '?'
        quotes.loc[:,'partialParentID'] = quotes['quoted_status'].apply(lambda x: x['id_str' + name_suffix])

        to_concat.append(quotes)

    quotes_of_replies = tweets[ tweets['is_quote_of_reply'] ]
    if len(quotes_of_replies) > 0:
        #for quotes of replies we don't know the root or the parent
        #the parent will be the reply or any retweets of the reply
        #the root can be tracked back using the parents in the tree
        quotes_of_replies.loc[:,'parentID'] = '?'
        quotes_of_replies.loc[:,'rootID'] = '?'
        quotes_of_replies.loc[:,'partialParentID'] = quotes_of_replies['quoted_status'].apply(lambda x: x['in_reply_to_status_id_str' + name_suffix])
        quotes_of_replies.loc[:,'actionType'] = 'quote'

        to_concat.append(quotes_of_replies)

    quotes_of_quotes = tweets[ tweets['is_quote_of_quote'] ]
    if len(quotes_of_quotes) > 0:
        #for quotes of quotes we don't know the parent or the root
        #the parent will be the first quote or any retweets of it
        #the root can be traced back through the parent tree
        quotes_of_quotes.loc[:,'parentID'] = '?'
        quotes_of_quotes.loc[:,'rootID'] = '?'
        quotes_of_quotes.loc[:,'partialParentID'] = quotes_of_quotes['quoted_status'].apply(lambda x: x['quoted_status_id_str'])
        quotes_of_quotes.loc[:,'actionType'] = 'quote'

        to_concat.append(quotes_of_quotes)

    orig_tweets = tweets[tweets['is_orig']]
    if len(orig_tweets) > 0:
        #for original tweets assign parent and root to be itself
        orig_tweets.loc[:,'actionType'] = 'tweet'
        orig_tweets.loc[:,'parentID'] = orig_tweets['nodeID']
        orig_tweets.loc[:,'rootID'] = orig_tweets['nodeID']
        orig_tweets.loc[:,'partialParentID'] = orig_tweets['nodeID']
        to_concat.append(orig_tweets)

    tweets = pd.concat(to_concat,ignore_index=True,sort=False)
        
    tweets = tweets[output_columns]

    print('Sorting...')
    tweets = tweets.sort_values("nodeTime").reset_index(drop=True)

    print('Reconstructing cascades...')
    tweets = full_reconstruction(tweets)

    #initialize info ID column with empty lists
    tweets['threadInfoIDs'] = [[] for i in range(len(tweets))]
    
    tweets = tweets.reset_index(drop=True)
    
    #get children of node
    def get_children(nodeID):

        children = tweets[tweets['parentID'] == nodeID]['nodeID']
        children = children[children.values != nodeID]
        
        return(children)


    #all comments on a post/comment mentioning a unit of information are also assigned that unit of information
    def add_info_to_children(nodeID,list_info=[]):

        infos = list(tweets[tweets['nodeID'] == nodeID]['informationIDs'].values[0])

        list_info = list_info.copy()

        children = get_children(nodeID)
        
        if len(children) > 0:

            list_info += infos
    
            if len(list_info) > 0 and len(children) > 1:
                #assign parents information ID list to all children
                tweets.loc[children.index.values,'threadInfoIDs'] = [list_info for i in range(len(children))]
            elif len(list_info) > 0 and len(children) == 1:
                #assign parents information ID list to single child
                tweets.at[children.index[0],'threadInfoIDs'] = list_info

            for child in children.values:
                #navigate further down the tree
                add_info_to_children(child,list_info)


    if get_info_ids:
        print('Adding information IDs to children...')
        #for each thread in data, propagate infromation IDs to children
        roots = tweets['rootID'].unique()
        for r,root in enumerate(roots):
            if root in tweets['nodeID'].values:
                add_info_to_children(root)
                if r % 50 == 0:
                    print('{}/{}'.format(r,len(roots)))

        tweets['informationIDs'] = tweets.apply(lambda x: list(set(x['informationIDs'] + x['threadInfoIDs'])),axis=1)
        tweets = tweets[tweets['informationIDs'].str.len() > 0]

    #tweets = tweets.drop("threadIn
        
    if get_info_ids:

        print('Expanding events...')
        #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
        s = tweets.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'informationID'
        tweets = tweets.drop(['informationIDs','partialParentID'], axis=1).join(s).reset_index(drop=True)
        
    tweets = tweets.drop('threadInfoIDs',axis=1)
    tweets = convert_timestamps(tweets)

    print('Done!')
    return tweets


    
def extract_github_data(fn='github_data.json',
                        info_id_fields=None,
                        keywords = [],
                        anonymized=False):

    json_data = load_json(fn)
    data = pd.DataFrame(json_data)

    get_info_ids = False
    if not info_id_fields is None or len(keywords) > 0:
        get_info_ids = True
    
    if anonymized:
        name_suffix = "_h"
        text_suffix = "_m"
    else:
        name_suffix = ""
        text_suffix = ""


    github_text_fields = {"PushEvent":["commits","message" + text_suffix],
                          "PullRequestEvent":["pull_request","body" + text_suffix],
                          "IssuesEvent":["issue","body" + text_suffix],
                          "CreateEvent":["description" + text_suffix],
                          "PullRequestReviewCommentEvent":["comment","body" + text_suffix],
                          "ForkEvent":["forkee","description" + text_suffix],
                          "IssueCommentEvent":["comment","body" + text_suffix],
                          "CommitCommentEvent":["comment","body" + text_suffix]}

        
    print('Extracting fields...')
    output_columns = ['nodeID', 'nodeUserID', 'actionType', 'nodeTime', 'platform']
    if get_info_ids:
        output_columns.append('informationIDs')

    
    if 'event' in data.columns:
        data.loc[:,'nodeTime'] = data['event'].apply(lambda x: x['created_at'])
        data.loc[:,'actionType'] = data['event'].apply(lambda x: x['type'])
        data.loc[:,'nodeUserID'] = data['event'].apply(lambda x: x['actor']['login' + name_suffix])
        data.loc[:,'nodeID'] = data['event'].apply(lambda x: x['repo']['name' + name_suffix])
    else:
        data.loc[:,'nodeUserID'] = data['actor'].apply(lambda x: x['login' + name_suffix])
        data.loc[:,'nodeID'] = data['repo'].apply(lambda x: x['name' + name_suffix])

        data.rename(columns={'created_at': 'nodeTime',
                                   'type':'actionType'}, inplace=True)
        
    data.loc[:,'platform'] = 'github'


    def get_text_field(row):

        if row['actionType'] not in github_text_fields.keys():
            return ''
    
        if row['actionType'] == 'PushEvent':
            text = ' '.join(c['message' + text_suffix] for c in row['payload']['commits'])
        else:
            text = row['payload']
        
            for f in github_text_fields[row['actionType']]:
                if f in text:
                    text = text[f]
                else:
                    text = ''
            
        return text

    
    if len(keywords) > 0:
        data.loc[:,'text_field'] = data.apply(get_text_field,axis=1)
        data = data.dropna(subset=['text_field'])
        data.loc[:,'informationIDs'] = data['text_field'].apply(lambda x: get_info_id_from_text([x], keywords))
        data = data.drop('text_field',axis=1)
    elif not info_id_fields == None: 
        data.loc[:,'informationIDs'] = pd.Series(data['socialsim_details'].apply(lambda x: list(itertools.chain.from_iterable([get_info_id_from_fields(m,info_id_fields) for m in x]))))
    
    events = data[output_columns]
    
    events = events[events.actionType.isin(['PullRequestEvent','IssuesEvent','CreateEvent','DeleteEvent','WatchEvent','ForkEvent',
                                            'PullRequestReviewCommentEvent','CommitCommentEvent','PushEvent','IssueCommentEvent'])]

    if get_info_ids:
        print('Expanding events...')    
        #expand lists of info IDs into seperate rows (i.e. an individual event is duplicated if it pertains to multiple information IDs)
        s = events.apply(lambda x: pd.Series(x['informationIDs']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'informationID'
        events = events.drop('informationIDs', axis=1).join(s).reset_index(drop=True)
        events = events.dropna(subset=['informationID'])
        
    events = convert_timestamps(events)

    events = events.drop_duplicates([c for c in events.columns if c != 'urlDomains'])
    
    print('Done!')
    return events

                        
def main():

    fn = 'twitter_data2.json'
    #fn = ['reddit_posts_data.json','reddit_comments_data.json']
    #fn = ['github_repo_data.json','github_events_data.json']
    
    #data = extract_reddit_data(fn, anonymized=True, keywords=['issue','recent','client','code','secure','version'])
    data = extract_twitter_data(fn, anonymized=False, keywords=['venzuela','maduro'])
    #data = extract_twitter_data(fn, anonymized=False, info_id_fields = ["entities.hashtags.text"])
    #data = extract_reddit_data(fn, anonymized=True, info_id_fields = ["extension.socialsim_keywords"])

    print(data)
    
if __name__ == "__main__":
    main()
