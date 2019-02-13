
import numpy as np
import pandas as pd

from ..measurements import MeasurementsBaseClass
from collections import Counter, defaultdict


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

    def order_of_spread_and_time_delta(self):
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
            if len(v) == 2:
                delta[k].append(pd.Timedelta(v[1] - v[0]).seconds)
            if len(v) == 3:
                delta[k].append(pd.Timedelta(v[1] - v[0]).seconds)
                delta[k].append(pd.Timedelta(v[2] - v[0]).seconds)

        time_df = pd.DataFrame(list(delta.items()), columns=[self.content, "time_delta"])
        new_df = pd.DataFrame(list(sorted_order.items()), columns=[self.content, "spread_order"])
        new_df = pd.merge(new_df, time_df, on=self.content)
        return new_df

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

        return pd.DataFrame(list(keyword_user_matrix.items()), columns=[self.content, "overlapping_users"])

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
        return pd.DataFrame(list(ranking_shares.items()), columns=[self.content, "ranking_shares"])

