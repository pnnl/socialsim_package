
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from collections import Counter, defaultdict
#from ..measurements import MeasurementsBaseClass

# TODO:
#   5. (Audience) Which platforms have the largest audience for the information?
#   6. (Speed) On which platforms does the information spread fastest?
#   9. (Audience) Do different platforms show similar temporal patterns of audience growth?


class CrossPlatformMeasurements():
    def __init__(self, dataset, platform="platform", parent_node_col="parentID", node_col="nodeID",
                 root_node_col="rootID", timestamp_col="nodeTime", user_col="nodeUserID", content_col="content",
                 log_file='cross_platform_measurements_log.txt'):
        super(CrossPlatformMeasurements, self).__init__()
#        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)

        self.dataset = dataset
        self.root_node_col = root_node_col
        self.parent_node_col = parent_node_col
        self.node_col = node_col
        self.timestamp_col = timestamp_col
        self.user_col = user_col
        self.platform = platform
        self.content = content_col
        self.content_df = pd.DataFrame()

    def order_of_spread_and_time_delta(self, time_granularity="S"):
        keywords_to_order = {}
        for index, row in self.dataset.iterrows():
            platform = row[self.platform]
            keywords = row[self.content]
            for k in keywords:
                try:
                    if platform not in keywords_to_order[k].keys():
                        keywords_to_order[k][platform] = index
                except KeyError:
                    keywords_to_order[k] = {platform: index}
        sorted_order = {}
        time_delta = {}
        for k, v in keywords_to_order.items():
            sorted_v = sorted(v.items(), key=lambda kv: kv[1])
            sorted_keys = [item[0] for item in sorted_v]
            sorted_order[k] = sorted_keys
            time = [item[1] for item in sorted_v]
            time_delta[k] = [self.dataset[self.timestamp_col].iloc[i] for i in time]

        delta = {}
        for k, v in time_delta.items():
            delta[k] = [0]
            if len(v) > 1:
                deltaTime = pd.Timedelta(v[1] - v[0]).seconds
                if time_granularity == "S":
                    delta[k].append(deltaTime)
                elif time_granularity == "M":
                    delta[k].append(deltaTime / 60.0)
                elif time_granularity == "H":
                    delta[k].append(deltaTime / 3600.0)
                else:
                    delta[k].append(pd.Timedelta(v[1] - v[0]).days)
            if len(v) > 2:
                deltaTime = pd.Timedelta(v[2] - v[0]).seconds
                if time_granularity == "S":
                    delta[k].append(deltaTime)
                elif time_granularity == "M":
                    delta[k].append(deltaTime / 60.0)
                elif time_granularity == "H":
                    delta[k].append(deltaTime / 3600.0)
                else:
                    delta[k].append(pd.Timedelta(v[2] - v[0]).days)

        time_df = pd.DataFrame(list(delta.items()), columns=[self.content, "time_delta"])
        new_df = pd.DataFrame(list(sorted_order.items()), columns=[self.content, "spread_order"])
        merged = pd.merge(new_df, time_df, on=self.content)
        if len(self.content_df) == 0:
            self.content_df = merged
        else:
            self.content_df = pd.merge(self.content_df, merged, on=self.content)

    def filter_common_users(self):
        all_platform_users = defaultdict(set)
        for index, row in self.dataset.iterrows():
            all_platform_users[row[self.user_col]].add(row[self.platform])
        users = {}
        for k, v in all_platform_users.items():
            # Remove users that only appear in 1 platform
            if len(v) > 1:
                users[k] = v
        return self.dataset.loc[self.dataset[self.user_col].isin(users.keys())]

    def select_data(self,data,nodes=[],communities=[]):
        
        if len(nodes) > 0:
            data = data[data[self.content].isin(nodes)]
        if len(communities) > 0:
            data = data[data[self.community].isin(communities)]
            
        return data

    
    def overlapping_users(self, nodes=[], communities=[]):

        data = self.select_data(self.dataset,nodes,communities)
        
        platforms = data[self.platform].unique()
                
        data['dummy'] = 1

        data = data.drop_duplicates(subset=[self.user_col,self.platform])

        cols = [self.user_col,
                self.platform,
                'dummy']
        index_cols = [self.user_col]
        if len(communities) > 0:
            cols = [self.user_col,
                    self.platform,
                    self.community,
                    'dummy']
            index_cols = [self.user_col,self.community]
            group_col = self.community
        if len(nodes) > 0:
            cols = [self.user_col,
                    self.platform,
                    self.content,
                    'dummy']
            index_cols = [self.user_col,self.content]
            group_col = 'content'

        user_platform = data[cols].pivot_table(index=index_cols,
                                               columns=self.platform, 
                                               values = 'dummy').fillna(0)

        user_platform = user_platform.astype(bool)


        def get_meas(grp):
            meas = np.zeros((len(platforms),len(platforms)))

            print(grp)

            for i,p1 in enumerate(platforms):
                for j,p2 in enumerate(platforms):
                
                    if p1 == p2:
                        x = 1.0
                    else:
                        x = (grp[p1] & grp[p2]).sum()
                        total = float(grp[p1].sum()) 
                        if total > 0:
                            x = x/total
                        else:
                            x = 0
                
                    meas[i][j] = x

            return meas

        if len(nodes) != 0 or len(communities) != 0:
            user_platform = user_platform.groupby(group_col)
            meas = user_platform.apply(get_meas).to_dict()
        else:
            meas = get_meas(user_platform)
        

        return(meas)

    def overlapping_users(self):
        keyword_user_matrix = {}

    def size_of_shares(self):
        content_to_size = {}
        for index, row in self.dataset.iterrows():
            platform = row[self.platform]
            keywords = row[self.content]
            for k in keywords:
                try:
                    content_to_size[k][platform] += 1
                except KeyError:
                    content_to_size[k] = Counter()
        ranking_shares = {}
        for k, v in content_to_size.items():
            sorted_v = sorted(v.items(), key=lambda kv: kv[1])
            sorted_keys = [item[0] for item in sorted_v]
            ranking_shares[k] = sorted_keys
        if len(self.content_df) == 0:
            self.content_df = pd.DataFrame(list(ranking_shares.items()), columns=[self.content, "ranking_shares"])
        else:
            self.content_df = pd.merge(self.content_df, pd.DataFrame(list(ranking_shares.items()),
                                                                     columns=[self.content, "ranking_shares"]),
                                       on=self.content)

    def temporal_share_correlation(self, time_granularity="D"):
        content_over_time = {}
        time_interval = set()
        for index, row in self.dataset.iterrows():
            time = row["nodeTime"]
            if time_granularity == "S":
                time = time
            elif time_granularity == "M":
                time = '{year}-{month:02}-{day}:{hour}:{min}'.format(year=time.year, month=time.month, day=time.day,
                                                                     hour=time.hour, min=time.minute)
            elif time_granularity == "H":
                time = '{year}-{month:02}-{day}:{hour}'.format(year=time.year, month=time.month, day=time.day,
                                                               hour=time.hour)
            else:
                time = '{year}-{month:02}-{day}'.format(year=time.year, month=time.month, day=time.day)
            time_interval.add(time)
            platform = row["platform"]
            keywords = row["content"]
            for k in keywords:
                try:
                    _ = content_over_time[k]
                except KeyError:
                    content_over_time[k] = {}
                time_dict = content_over_time[k]
                try:
                    _ = time_dict[time]
                except KeyError:
                    time_dict[time] = {}
                plat_dict = time_dict[time]
                try:
                    _ = plat_dict[platform]
                except KeyError:
                    plat_dict[platform] = 0
                plat_dict[platform] += 1
        content_to_correlation = {}
        content_to_time_series = defaultdict()
        sort_time = sorted(time_interval)
        for k, v in content_over_time.items():
            content_to_time_series[k] = {"twitter": [], "reddit": [], "github": []}
            for t in sort_time:
                try:
                    plats = v[t]
                    for p in ["twitter", "reddit", "github"]:
                        try:
                            content_to_time_series[k][p].append(plats[p])
                        except KeyError:
                            content_to_time_series[k][p].append(0)
                except KeyError:
                    content_to_time_series[k]['twitter'].append(0)
                    content_to_time_series[k]['github'].append(0)
                    content_to_time_series[k]['reddit'].append(0)
        for k, v in content_to_time_series.items():
            tg_corr = pearsonr(np.array(v["twitter"]), np.array(v["github"]))
            tr_corr = pearsonr(np.array(v["twitter"]), np.array(v["reddit"]))
            rg_corr = pearsonr(np.array(v["reddit"]), np.array(v["github"]))
            content_to_correlation[k] = [[1.0, tg_corr[0], tr_corr[0]], [tg_corr[0], 1.0, rg_corr[0]],
                                         [tr_corr[0], rg_corr[0], 1.0]]

        if len(self.content_df) == 0:
            self.content_df = pd.DataFrame(list(content_to_correlation.items()),
                                           columns=["content", "temporal_share_correlation"])
        else:
            self.content_df = pd.merge(self.content_df, pd.DataFrame(list(content_to_correlation.items()),
                                                                     columns=["content", "temporal_share_correlation"]),
                                       on=self.content)

    def lifetime_of_spread(self):
        keywords_to_order = {}
        for index, row in self.dataset.iterrows():
            platform = row["platform"]
            keywords = row["content"]
            time = row["nodeTime"]
            for k in keywords:
                try:
                    if platform not in keywords_to_order[k].keys():
                        keywords_to_order[k][platform] = [time]
                    else:
                        keywords_to_order[k][platform].append(time)
                except KeyError:
                    keywords_to_order[k] = {platform: [time]}
        content_to_lifetime = {}
        for k, v in keywords_to_order.items():
            lifetime_rank = {}
            for plat, times in v.items():
                lifetime_rank[plat] = pd.Timedelta(times[-1] - times[0])
            sorted_v = sorted(lifetime_rank.items(), key=lambda kv: kv[1])
            sorted_keys = [item[0] for item in sorted_v]
            content_to_lifetime[k] = sorted_keys
        if len(self.content_df) == 0:
            self.content_df = pd.DataFrame(list(content_to_lifetime.items()), columns=["content", "lifetime"])
        else:
            self.content_df = pd.merge(self.content_df, pd.DataFrame(list(content_to_lifetime.items()),
                                                                     columns=["content", "lifetime"]),
                                       on=self.content)
