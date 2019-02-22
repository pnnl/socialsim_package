# from cross_platform import CrossPlatformMeasurements
import pandas as pd
import pickle
import pull_from_mongo
import socialsim as ss
import time
import warnings
warnings.filterwarnings("ignore")


def load_data():
    reddit_data = pull_from_mongo.extract_reddit_data_from_mongo("Reddit_CVE")
    reddit_data["platform"] = "reddit"
    twitter_data = pull_from_mongo.extract_twitter_data_from_mongo("Twitter_CVE")
    twitter_data["platform"] = "twitter"
    github_data = pull_from_mongo.extract_github_data_from_mongo("GitHub_CVE")
    github_data["platform"] = "github"
    all_plats = pd.concat([reddit_data, twitter_data, github_data], sort=False)
    print(list(all_plats))
    all_plats = all_plats.sort_values('nodeTime')

    return all_plats


def get_reddit_activity(df):
    df = df.loc[df["platform"] == "reddit"]
    df["nodeID"] = df["nodeID"].apply(lambda x: x[3:])
    df["parentID"] = df["parentID"].apply(lambda x: x[3:])
    df["rootID"] = df["rootID"].apply(lambda x: x[3:])
    print(df[["nodeID", "parentID", "rootID"]].head())
    new_df_list = []
    for idx, cascade_df in df.groupby("rootID"):
        # print(len(cascade_df))
        cascade_df["audience"] = 1
        parentids = cascade_df["parentID"].tolist()
        leafs = cascade_df.loc[~cascade_df["nodeID"].isin(parentids)]
        _ = calculate_activity(leafs, cascade_df, idx)
        # df = pd.merge(df, cascade_df[["nodeID","audience"]], on="nodeID")
        new_df_list.append(cascade_df)
    return pd.concat(new_df_list)


def calculate_activity(leafs, cascade, rootid):
    # print(leafs.head())
    if len(leafs) == 0 or leafs["nodeID"].values[0] == rootid:
        return cascade
    else:
        parent_list = set(leafs["parentID"].tolist())
        nodes = cascade.loc[cascade["nodeID"].isin(parent_list), "nodeID"].tolist()
        for n in nodes:
            cascade.loc[cascade["nodeID"] == n, "audience"] += cascade.loc[cascade["parentID"] == n, "audience"].sum()
        parents = cascade.loc[cascade["nodeID"].isin(parent_list)]
        # print(cascade.head())
        calculate_activity(parents, cascade, rootid)


def get_github_activity(df):
    df.loc[df["platform"] == "github", "audience"] = 1
    return df


#~~~~~~ PREPROCESSING DATA ~~~~~~~~~~~~#

data = load_data()

data = data[data['informationID'].str.len() > 0]
data = data.sample(frac=0.01, random_state=27)
# Define communities and node list subsets
nodes = ['CVE-2016-2216', 'CVE-2016-7099', 'CVE-2016-2216',
         'CVE-2016-1019', 'CVE-2014-9390', 'CVE-2017-7533', 'CVE-2017-5638']
commA = ['CVE-2016-2216', 'CVE-2016-7099', 'CVE-2016-2216']
commB = ['CVE-2016-1019', 'CVE-2014-9390', 'CVE-2017-7533', 'CVE-2016-2216']

# Format content df
s = data.apply(lambda x: pd.Series(x['informationID']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'informationID'

data = data.drop('informationID', axis=1).join(s).reset_index(drop=True)
data = data.dropna(subset=['nodeUserID'])

# Compute audience per content
# start = time.time()
# new_data = get_reddit_activity(data)
# end = time.time()
# print(end-start)
# print(list(new_data))
# data = get_github_activity(new_data)
# print(new_data["audience"].value_counts())

# Format community df
comm_df = []
for name, comm in zip(["A", "B"],[commA, commB]):
    temp = data.loc[data["informationID"].isin(comm)]
    temp["community"] = name
    comm_df.append(temp)

community_df = pd.concat(comm_df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# cpm = ss.CrossPlatformMeasurements(data, configuration={})
# print("----------------------------------------")
# print("NODE LEVEL")
# print("----------------------------------------")
# print("***** 1 ORDER OF SPREAD")
# meas = cpm.order_of_spread(nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 2 TIME DELTA")
# meas = cpm.time_delta(nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 3 OVERLAPPING USERS")
# meas = cpm.overlapping_users(nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 4 SIZE OF SHARES")
# meas = cpm.size_of_shares(nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 5 SIZE OF AUDIENCE")
# meas = cpm.size_of_audience(nodes=nodes)
# for k, v in meas.items():
#     print(k,v)
# print("***** 6 SPEED OF SPREAD")
# meas = cpm.speed_of_spread(nodes=nodes)
# for k, v in meas.items():
#     print(k,v)
# print("***** 7 LIFETIME")
# meas = cpm.lifetime_of_spread(nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 8 SHARE CORRELATION")
# meas = cpm.temporal_correlation(measure="share", nodes=nodes)
# for k, v in meas.items():
#     print(k, v)
# print("***** 9 AUDIENCE CORRELATION")
# meas = cpm.temporal_correlation(measure="audience", nodes=nodes)
# for k, v in meas.items():
#     print(k, v)


cpm = ss.CrossPlatformMeasurements(data, configuration={})
# print("----------------------------------------")
# print("POPULATION LEVEL")
# print("----------------------------------------")
# print("***** 10 ORDER OF SPREAD")
# meas = cpm.order_of_spread()
# for k, v in meas.items():
#     print(k, v)
# print("***** 11 TIME DELTA")
# meas = cpm.time_delta()
# for k in meas:
#     print(k)
# print("***** 12 OVERLAPPING USERS")
# meas = cpm.overlapping_users()
# for k in meas:
#     print(k)
# print("***** 13 SIZE OF SHARES")
# meas = cpm.size_of_shares()
# for k in meas:
#     print(k)
# print("***** 14 SPEED OF SPREAD")
# meas = cpm.speed_of_spread()
# for k in meas:
#     print(k)
# print("***** 15 LIFETIME")
# meas = cpm.lifetime_of_spread()
# for k in meas:
#     print(k)
# print("***** 16 SHARE CORRELATION")
# meas = cpm.correlation_of_information(measure="share")
# for k in meas:
#     print(k)
# print("***** 17 AUDIENCE CORRELATION")
# meas = cpm.correlation_of_information(measure="audience")
# for k in meas:
#     print(k)
# print("***** 18 LIFETIME CORRELATION")
# meas = cpm.correlation_of_information(measure="lifetime")
# for k in meas:
#     print(k)
# print("***** 19 SPEED CORRELATION")
# meas = cpm.correlation_of_information(measure="speed")
# for k in meas:
#     print(k)

# print(list(community_df))
cpm = ss.CrossPlatformMeasurements(data, configuration={}, communities=community_df, community_list="all")
print("----------------------------------------")
print("COMMUNITY LEVEL")
print("----------------------------------------")
print("***** 10 ORDER OF SPREAD")
meas = cpm.order_of_spread()
for k, v in meas.items():
    print(k, v)
print("***** 11 TIME DELTA")
meas = cpm.time_delta()
for k, v in meas.items():
    print(k, v)
print("***** 12 OVERLAPPING USERS")
meas = cpm.overlapping_users()
for k, v in meas.items():
    print(k, v)
print("***** 13 SIZE OF SHARES")
meas = cpm.size_of_shares()
for k, v in meas.items():
    print(k, v)
print("***** 14 SPEED OF SPREAD")
meas = cpm.speed_of_spread()
for k, v in meas.items():
    print(k, v)
print("***** 15 LIFETIME")
meas = cpm.lifetime_of_spread()
for k, v in meas.items():
    print(k, v)
print("***** 16 SHARE CORRELATION")
meas = cpm.correlation_of_information(measure="share")
print(meas)
print("***** 17 AUDIENCE CORRELATION")
meas = cpm.correlation_of_information(measure="audience")
print(meas)
print("***** 18 LIFETIME CORRELATION")
meas = cpm.correlation_of_information(measure="lifetime")
print(meas)
print("***** 19 SPEED CORRELATION")
meas = cpm.correlation_of_information(measure="speed")
print(meas)
