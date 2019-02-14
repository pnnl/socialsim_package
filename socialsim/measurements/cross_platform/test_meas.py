from cross_platform import CrossPlatformMeasurements
import pandas as pd
import pickle

with open('train_data.pkl','rb') as f:
    data = pickle.load(f)
print(data['content'].head())

print(data)

data = data[data['content'].str.len() > 0]
data = data.sample(frac=0.01,random_state=27)

print(data)

s = data.apply(lambda x: pd.Series(x['content']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'content'

print(s)

data = data.drop('content', axis=1).join(s).reset_index(drop=True)

data = data.dropna(subset=['nodeUserID'])

print(data)


cpm = CrossPlatformMeasurements(data)


print(cpm.overlapping_users2())

#print(cpm.overlapping_users2(nodes=['CVE-2016-2216','CVE-2016-7099','CVE-2016-2216',
#                                    'CVE-2016-1019','CVE-2014-9390','CVE-2017-7533']))
