import pandas as pd

from pymongo import MongoClient
from pandas.io.json import json_normalize


def convert_datetime(dataset, verbose=None):
    """
    Description:

    Input:

    Output:
    """

    if verbose:
        print('Converting strings to datetime objects...', end='', flush=True)

    try:
        dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='s')
    except:
        try:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'], unit='ms')
        except:
            dataset['nodeTime'] = pd.to_datetime(dataset['nodeTime'])

    if verbose:
        print(' Done')

    return dataset


def extract_github_data_from_mongo(collection, database="Jun19-train", extras=None):
    mongo_client = MongoClient()

    if "CVE" in collection:
        select = {"id_h": 1,
                  "actor.id_h": 1,
                  "created_at": 1,
                  "socialsim_details.extension.socialsim_keywords": 1,
                  "repo.id_h": 1,
                  "_id": 0}
        if extras is not None:
            select.update(extras)
        coll = mongo_client[database][collection + "_events"]
        events = coll.find({}, select)
        events = json_normalize(list(events))
        events.rename(columns={'id_h': "nodeID", 'actor.id_h': "nodeUserID",
                               'created_at': "nodeTime", 'socialsim_details': 'informationID',
                               "repo.id_h": "rootID"}, inplace=True)
        events = convert_datetime(events)
        # events['nodeTime'] = pd.to_datetime(events['nodeTime'], infer_datetime_format=True)
        events["informationID"] = events["informationID"].apply(lambda x: x[0]['extension']['socialsim_keywords'])

        events.drop_duplicates(subset='nodeID', inplace=True)

        coll = mongo_client[database][collection + "_repos"]
        repos = coll.find({}, select)
        repos = json_normalize(list(repos))
        repos.rename(columns={'id_h': "nodeID", 'actor.id_h': "nodeUserID",
                              'created_at': "nodeTime", 'socialsim_details': 'informationID',
                              "repo.id_h": "rootID"}, inplace=True)
        # repos['nodeTime'] = pd.to_datetime(repos['nodeTime'], infer_datetime_format=True)
        repos = convert_datetime(repos)
        repos["informationID"] = repos["informationID"].apply(lambda x: x[0]['extension']['socialsim_keywords'])
        repos.drop_duplicates(subset='nodeID', inplace=True)
        return events.append(repos).reset_index(drop=True)


def extract_twitter_data_from_mongo(collection, database="Jun19-train", extras=None):
    mongo_client = MongoClient()
    coll = mongo_client[database][collection]
    if "URL" in collection:
        exploit = 'extension.socialsim_urls_m'
        renamed = "informationID"
    else:
        exploit = 'extension.socialsim_keywords'
        renamed = "informationID"
    select = {'id_h': 1,
              'created_at': 1,
              'username_h': 1,
              'in_reply_to_user_id_h': 1,
              exploit: 1,
              '_id': 0}
    if extras is not None:
        select.update(extras)
    data = coll.find({}, select)
    data = json_normalize(list(data))
    data.rename(columns={'id_h': "nodeID", 'username_h': "nodeUserID", "in_reply_to_user_id_h": "rootID",
                         'created_at': "nodeTime", exploit: renamed}, inplace=True)
    # data['nodeTime'] = pd.to_datetime(data['nodeTime'], infer_datetime_format=True)
    data = convert_datetime(data)
    data["rootID"].apply(lambda x: data["nodeID"] if x is None else x)
    data.drop_duplicates(subset='nodeID', inplace=True)
    return data


def extract_reddit_data_from_mongo(collection, database="Jun19-train", extras=None):
    mongo_client = MongoClient()

    if "CVE" in collection:
        coll = mongo_client[database][collection + "_posts"]
        select = {'id_h': 1,
                  'author_h': 1,
                  'created_date': 1,
                  'extension.socialsim_keywords': 1,
                  '_id': 0}
        if extras is not None:
            select.update(extras)
        data = coll.find({}, select)
        data = json_normalize(list(data))
        data.rename(columns={'id_h': "nodeID", 'author_h': "nodeUserID",
                             'created_date': "nodeTime", 'extension.socialsim_keywords': 'informationID'}, inplace=True)

        data['nodeID'] = data['nodeID'].apply(lambda x: 't3_{}'.format(x))
        data["rootID"] = data["nodeID"]
        data['parentID'] = ""
        # data['nodeTime'] = pd.to_datetime(data['nodeTime'], infer_datetime_format=True)
        data = convert_datetime(data)

        data.drop_duplicates(subset='nodeID', inplace=True)

        coll = mongo_client[database][collection + "_comments"]
        select = {'id_h': 1,
                  'author_h': 1,
                  'created_date': 1,
                  'parent_id_h': 1,
                  'link_id_h': 1,
                  'extension.socialsim_keywords': 1,
                  '_id': 0}
        if extras is not None:
            select.update(extras)
        comments = coll.find({}, select)
        comments = json_normalize(list(comments))
        comments.rename(
            columns={'id_h': "nodeID", 'author_h': "nodeUserID", "parent_id_h": "parentID", "link_id_h": "rootID",
                     'created_date': "nodeTime", 'extension.socialsim_keywords': 'informationID'}, inplace=True)

        comments['nodeID'] = comments['nodeID'].apply(lambda x: 't1_{}'.format(x))
        # comments['nodeTime'] = pd.to_datetime(comments['nodeTime'], infer_datetime_format=True)
        comments = convert_datetime(comments)
        comments.drop_duplicates(subset='nodeID', inplace=True)
        print(len(comments))
        return data.append(comments).reset_index(drop=True)

    if "URL" in collection:
        coll = mongo_client[database][collection]
        data = coll.find({}, {'id_h': 1,
                              'author_h': 1,
                              'created_date': 1,
                              'extension.socialsim_keywords': 1,
                              'extension.socialsim_urls_m': 1,
                              'parent_id_h': 1,
                              'link_id_h': 1,
                              '_id': 0})
        data = json_normalize(list(data))
        data.rename(
            columns={'id_h': 'nodeID', 'author_h': 'nodeUserID', 'parent_id_h': 'parentID', 'link_id_h': 'rootID',
                     'created_date': "nodeTime", 'extension.socialsim_urls_m': 'informationID'}, inplace=True)
        # data['nodeTime'] = pd.to_datetime(data['nodeTime'], infer_datetime_format=True)
        data = convert_datetime(data)
        data.drop_duplicates(subset='nodeID', inplace=True)
        return data
