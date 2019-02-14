
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from collections import Counter, defaultdict
from ..measurements import MeasurementsBaseClass

# TODO:
#   5. (Audience) Which platforms have the largest audience for the information?
#   6. (Speed) On which platforms does the information spread fastest?
#   9. (Audience) Do different platforms show similar temporal patterns of audience growth?


class CrossPlatformMeasurements(MeasurementsBaseClass):
    def __init__(self, dataset, configuration, platform="platform", parent_node_col="parentID", node_col="nodeID",
                 root_node_col="rootID", timestamp_col="nodeTime", user_col="nodeUserID", content_col="content",
                 log_file='cross_platform_measurements_log.txt'):
        super(CrossPlatformMeasurements, self).__init__(dataset, configuration, log_file=log_file)

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

    def overlapping_users(self):
        keyword_user_matrix = {}
        tg_total = len(set(self.dataset.loc[(self.dataset[self.platform] == "twitter") |
                                            (self.dataset[self.platform] == "github"), self.user_col].tolist()))
        tr_total = len(set(self.dataset.loc[(self.dataset[self.platform] == "twitter") |
                                            (self.dataset[self.platform] == "reddit"), self.user_col].tolist()))
        rg_total = len(set(self.dataset.loc[(self.dataset[self.platform] == "reddit") |
                                            (self.dataset[self.platform] == "github"), self.user_col].tolist()))
        # content from all users
        all_keywords_list = self.dataset[self.content].tolist()
        all_keywords_set = set()
        for key in all_keywords_list:
            for k in key:
                all_keywords_set.add(k)
        common_users = self.filter_common_users()
        common_users = common_users[[self.platform, self.user_col, self.content]]
        common_users = common_users.loc[common_users[self.content].str.len() != 0]
        # content from only cross-platform users
        keyword_list = common_users[self.content].tolist()
        common_users[self.content] = common_users[self.content].apply(lambda x: " ".join(x))
        keyword_set = set()
        for key in keyword_list:
            for k in key:
                keyword_set.add(k)
        for k in keyword_set:
            twitter_users = set(common_users.loc[(common_users[self.platform] == "twitter") &
                                                 (common_users[self.content].str.contains(k)), self.user_col].tolist())
            github_users = set(common_users.loc[(common_users[self.platform] == 'github') &
                                                (common_users[self.content].str.contains(k)), self.user_col].tolist())
            reddit_users = set(common_users.loc[(common_users[self.platform] == 'reddit') &
                                                (common_users[self.content].str.contains(k)), self.user_col].tolist())
            tr_users = len(twitter_users.intersection(reddit_users))
            tg_users = len(twitter_users.intersection(github_users))
            rg_users = len(reddit_users.intersection(github_users))
            if len(twitter_users.union(github_users)) == 0:
                tg_total_users = 0
            else:
                tg_total_users = tg_users / float(tg_total)
            if len(twitter_users.union(reddit_users)) == 0:
                tr_total_users = 0
            else:
                tr_total_users = tr_users / float(tr_total)
            if len(reddit_users.union(github_users)) == 0:
                rg_total_users = 0
            else:
                rg_total_users = rg_users / float(rg_total)

            matrix = [[1.0, tg_total_users, tr_total_users],
                      [tg_total_users, 1.0, rg_total_users],
                      [tr_total_users, rg_total_users, 1.0]]
            keyword_user_matrix[k] = np.array(matrix)
        # content from only single platform users
        single_platform_keywords = all_keywords_set.difference(keyword_set)
        for k in single_platform_keywords:
            matrix = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
            keyword_user_matrix[k] = np.array(matrix)
        if len(self.content_df) == 0:
            self.content_df = pd.DataFrame(list(keyword_user_matrix.items()),
                                           columns=[self.content, "overlapping_users"])
        else:
            self.content_df = pd.merge(self.content_df, pd.DataFrame(list(keyword_user_matrix.items()),
                                                                     columns=[self.content, "overlapping_users"]),
                                       on=self.content)

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
