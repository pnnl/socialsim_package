import pandas as pd
import json

import socialsim as ss

import glob
import re
import datetime
import numpy as np
import uuid

import pprint

import time

    
def add_new_users(data, hist_data, previous_users = []):
    
    print('Adding new users...')

    #get the fraction of new users to all users in the historical data
    new_users = hist_data[~hist_data['nodeUserID'].isin(previous_users)]
    new_users = new_users['nodeUserID'].nunique()
    users = hist_data['nodeUserID'].nunique()
    new_user_rate = float(new_users) / float(users)    

    #get unique users in the sampled data
    new_users = data.drop_duplicates('nodeUserID')
    new_users['new_user_rate'] = new_user_rate
        
    #randomly select users to replace based on desired rate of new users
    new_users['replacement'] = new_users['nodeUserID']
    new_users['random'] = np.random.uniform(0,1,len(new_users))
    
    idx = new_users['random'] <= new_users['new_user_rate']
    new_users['replace'] = idx

    #randomly generate user IDs for the new users
    new_users = new_users[idx]
    new_users['replacement'] = [str(uuid.uuid4()) for i in range(len(new_users))]

    #rename users in the original sampled data using the new usernames
    data = data.merge(new_users[['nodeUserID','replacement']],on='nodeUserID',how='left')
    data['replacement'] = data['replacement'].fillna(data['nodeUserID'])
    data = data.drop('nodeUserID',axis=1)
    data = data.rename(columns={'replacement':'nodeUserID'})

    new_users = data[~data['nodeUserID'].isin(previous_users + list(hist_data['nodeUserID'].unique()))]
    new_users = new_users['nodeUserID'].nunique()
    users = data['nodeUserID'].nunique()
    new_user_rate = float(new_users) / float(users)

    for col in ['new_user_rate','random','replacement']:
        if col in data.columns:
            data = data.drop(col,axis=1)

    return(data)
        
    
def interevent_times(data, elapsed):
    #sample inter-event times from the historical data
    #draw enough samples to cover the simulation time period

    print('Sampling inter-event times...')
    #calculate inter-event times
    delta = data['nodeTime'].diff().dropna()
    
    mean_delta = delta.mean()

    n_samples = int(2.0*(elapsed/mean_delta))

    sample = delta.sample(n_samples,replace=True)
    
    #increase size of sample until it is long enough
    while sample.sum() < elapsed:
        n_samples *= 2.0
        n_samples = int(n_samples)
        sample = delta.sample(n_samples,replace=True)
        
    #cut off extra data beyond the end of the simulation period
    sample = sample[sample.cumsum() < elapsed]
    
    return(sample)

def fix_parent_relationships(sampled_df):

    #if parents occur after children in the randomly sampled data, flip their timestamps

    print('Fixing parent relationships...')
    swap_parents = True
    counter = 0
    while swap_parents:

        orig = sampled_df.copy().reset_index(drop=True)
        sampled_df = sampled_df.reset_index()

        #merge data with itself to get parent information
        parents = sampled_df[['index','nodeID','parentID']].merge(sampled_df[['index','nodeID']],how='left',
                                                                  left_on='parentID',right_on='nodeID',suffixes=('_node','_parent'))

        #find pairs where the parent occurs after the child
        parents = parents.dropna()
        parents = parents[parents['index_parent'] > parents['index_node']]
        parents = parents.drop_duplicates(subset=['parentID'])
        parents['index_parent'] = parents['index_parent'].astype(int)

        #create dictionary mapping the index of the child to the index of the parent and vice versa
        index_dict = pd.Series(parents.index_node.values,index=parents.index_parent).to_dict()
        index_dict2 = pd.Series(parents.index_parent.values,index=parents.index_node).to_dict()
        index_dict.update(index_dict2)

        #swap indices of parents and children
        sampled_df = sampled_df.rename(index=index_dict).sort_index()
        sampled_df = sampled_df.drop('index',axis=1)
        sampled_df = sampled_df.reset_index(drop=True)
        
        #if nothing was swapped this time, stop swapping
        if (orig['nodeID'] == sampled_df['nodeID']).all():
            swap_parents = False
        counter += 1
        
    return(sampled_df)
        
def sample_from_historical_data(grp, info, plat, min_time, max_time, start_time,end_time,
                                previous_hist = pd.DataFrame(columns=['nodeUserID','informationID']),
                                new_users=True):

    #add fake starting and ending event to create inter-event times from the beginning to the first
    #event and from the last event to the end of the data
    fake_start_event = pd.DataFrame({'informationID':[info],'nodeTime':[min_time],'nodeID':['-'],
                                 'nodeUserID':['-'],'parentID':['-'],'rootID':['-'],'platform':[plat],
                                     'actionType':['-']})
    fake_end_event = pd.DataFrame({'informationID':[info],'nodeTime':[max_time],'nodeID':['-'],
                                   'nodeUserID':['-'],'parentID':['-'],'rootID':['-'],'platform':[plat],
                                   'actionType':['-']})
    grp = pd.concat([fake_start_event,grp,fake_end_event],sort=False)

    #sample inter-event_times from the data
    delta_times = interevent_times(grp,end_time - start_time).reset_index(drop=True)
    #convert the inter-event times to actual times from the start of the simulation
    times = start_time + delta_times.cumsum()
    
    #remove fake events
    grp = grp[grp['nodeID'] != '-']
    #sample random events from the historical data
    sampled_df = grp.sample(len(delta_times),replace=True).reset_index(drop=True)

    #if duplicate events have been sampled, rename their ID fields
    sampled_df['counter'] = (sampled_df.groupby(['nodeID','parentID']).cumcount()+1).astype(str)
    sampled_df['nodeID'] = sampled_df['nodeID'] + '-' + sampled_df['counter']
    sampled_df['parentID'] = sampled_df['parentID'] + '-' + sampled_df['counter']
    sampled_df = sampled_df.drop('counter',axis=1)

    #if parents occur after children, swap them
    sampled_df = fix_parent_relationships(sampled_df)

    #assign the previously sampled times to the dataframe
    sampled_df['nodeTime'] = times
    sampled_df['nodeTime'] = pd.to_datetime(sampled_df['nodeTime'])
    sampled_df = sampled_df.sort_values('nodeTime')
    
    #replace some users with new users that haven't been observed before
    if new_users:
        previous_users = list(previous_hist[previous_hist['informationID'] == info]['nodeUserID'].unique())
        sampled_df = add_new_users(sampled_df,grp,previous_users = previous_users)

    sampled_df['nodeTime'] = pd.to_datetime(sampled_df['nodeTime'])
    sampled_df = sampled_df.sort_values('nodeTime')

    return(sampled_df)

def main():

    path = './'
    n_runs = 1
    simulation_periods = [['2019-02-01','2019-02-15'],
                          ['2019-02-08','2019-02-22'],
                          ['2019-02-15','2019-03-01'],
                          ['2019-02-22','2019-03-01']]


    #files are in weekly subsets, e.g. venezuela_v2_extracted_twitter_2019-02-01_2019-02-08.json
    all_files = glob.glob(path + 'venezuela_v2_extracted*.json')

    #extract dates and platforms from file names
    date_re = '(20\d\d-\d\d-\d\d)_(20\d\d-\d\d-\d\d)'
    dates = [re.search(date_re,fn) for fn in all_files]
    start_dates = [d.group(1) for d in dates]
    end_dates = [d.group(2) for d in dates]
    platforms = [re.search('twitter|youtube',fn).group(0) for fn in all_files]

    #create data frame with files, dates, and platforms
    fn_df = pd.DataFrame({'fn':all_files,
                          'start':start_dates,
                          'end':end_dates,
                          'platform':platforms})

    fn_df['start'] = pd.to_datetime(fn_df['start'])
    fn_df['end'] = pd.to_datetime(fn_df['end'])

    fn_df = fn_df.sort_values('start')

    #loop over simulation periods
    for sim_period in simulation_periods:
        #start and end time of the simulation
        start = pd.to_datetime(sim_period[0])
        end = pd.to_datetime(sim_period[1])

        #select files to sample from based on dates (same length as simulation period, just before simulation period)
        sim = fn_df[ (fn_df['start'] >= start) & (fn_df['start'] < end) ]
        hist = fn_df[ (fn_df['start'] < start) & (fn_df['start'] >= start - (end - start)) ]
        previous = fn_df[ (fn_df['start'] < start - (end - start)) ]
        
        print(start,end)
        print('Historical Data to Sample From')
        print(hist)
        print('Prior Data to Track Users From')
        print(previous)
        
        previous_history_data = list(previous['fn'].values)
        history_data = list(hist['fn'].values)

        #load data
        hist = []
        for data in history_data:
            hist.append(ss.load_data(data, ignore_first_line=False, verbose=False))
        hist = pd.concat(hist)

        hist = hist.sort_values('nodeTime')

        previous_hist = []
        for data in previous_history_data:
            previous_hist.append(ss.load_data(data, ignore_first_line=False, verbose=False))
        previous_hist = pd.concat(previous_hist)
        previous_hist = previous_hist[['nodeUserID','informationID']].drop_duplicates()

        #multiple runs of the baseline sampling
        for i in range(n_runs):
            dfs = []
            # for each platform and information ID
            for (plat,info),grp in hist.groupby(['platform','informationID']):

                print(plat,info)

                starting = time.time()
                sampled_df = sample_from_historical_data(grp, info, plat,
                                                         hist['nodeTime'].min(), hist['nodeTime'].max(),
                                                         start,end,
                                                         previous_hist = previous_hist,
                                                         new_users=True)
                ending = time.time()
                elapsed = (ending - starting)/60.0
                print(f'Time elapsed: {elapsed} minutes')

                dfs.append(sampled_df)

            baseline = pd.concat(dfs).reset_index(drop=True)

            #save generated baseline
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            baseline.to_json(f'baseline_{start_str}_{end_str}_{i}.json',orient='records',lines=True)
    
if __name__ == '__main__':
   main()
