from cross_platform import CrossPlatformMeasurements
import pandas as pd
import pickle
import pull_from_mongo


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


data = load_data()

data = data[data['content'].str.len() > 0]
data = data.sample(frac=0.01, random_state=27)
nodes = ['CVE-2016-2216', 'CVE-2016-7099', 'CVE-2016-2216',
         'CVE-2016-1019', 'CVE-2014-9390', 'CVE-2017-7533', 'CVE-2017-5638']

s = data.apply(lambda x: pd.Series(x['content']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'content'

data = data.drop('content', axis=1).join(s).reset_index(drop=True)
print(len(data))
data = data.dropna(subset=['nodeUserID'])
print(len(data))

cpm = CrossPlatformMeasurements(data)
# print("OVERLAPPING USERS")
# meas = cpm.overlapping_users()
# print(meas)
# meas = cpm.overlapping_users(nodes=nodes)
# for k ,v in meas.items():
#     print(k, v)
# print("ORDER OF SPREAD")
# meas = cpm.order_of_spread(nodes=nodes)
# for k ,v in meas.items():
#     print(k, v)
# print("TIME DELTA")
# meas = cpm.time_delta(nodes=nodes)
# for k ,v in meas.items():
#     print(k, v)
# print("SIZE OF SHARES")
# meas = cpm.size_of_shares()
# for k in meas:
#     print(k)
# meas = cpm.size_of_shares(nodes=nodes)
# for k ,v in meas.items():
#     print(k, v)
print("SHARE CORRELATION")
meas = cpm.temporal_share_correlation()
for k in meas:
    print(k)
meas = cpm.temporal_share_correlation(nodes=nodes)
for k ,v in meas.items():
    print(k, v)
print("LIFETIME")
meas = cpm.lifetime_of_spread()
for k ,v in meas.items():
    print(k, v)
meas = cpm.lifetime_of_spread(nodes=nodes)
for k ,v in meas.items():
    print(k, v)
