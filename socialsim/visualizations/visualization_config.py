measurement_plot_params = {

    ### community

    "community_burstiness": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Burstiness",
        "plot": ['bar']
    },

    "community_contributing_users": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Proportion of Users Contributing",
        "plot": ['bar']
    },

    "community_event_proportions": {
        "data_type": "dict_DataFrame",
        "x_axis": "Event Type",
        "y_axis": "Event Proportion",
        "plot": ['bar'],
        "plot_keys": "community"
    },

    "community_geo_locations": {
        "data_type": "dict_DataFrame",
        "x_axis": "Country",
        "y_axis": "Number of Events",
        "plot": ['bar'],
        "plot_keys": "community"
    },

    "community_issue_types": {  # result None type
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Issues",
        "plot": ['multi_time_series'],
        "plot_keys": "community"

    },

    "community_num_user_actions": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Mean Number of User Actions",
        "hue": "Key",
        "plot": ['time_series'],
        "plot_keys": "community_subsets"
    },
    #

    'community_user_account_ages': {
        "data_type": "dict_Series",
        "x_axis": "User Account Age",
        "y_axis": "Number of Actions",
        "plot": ['hist'],
        "plot_keys": "community"
    },

    'community_user_burstiness': {
        "data_type": "dict_Series",
        "x_axis": "User Burstiness",
        "y_axis": "Number of Users",
        "plot": ['hist'],
        "plot_keys": "community"
    },

    #
    "community_gini": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Gini Scores",
        "plot": ['bar']
    },

    "community_palma": {
        "data_type": "dict",
        "x_axis": "Community",
        "y_axis": "Palma Scores",
        "plot": ['bar']
    },

    # repo
    #

    "content_contributors": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Contributors",
        "plot": ['time_series'],
        "plot_keys": "content"
    },

    "content_diffusion_delay": {
        "data_type": "dict_Series",
        "x_axis": "Diffusion Delay",
        "y_axis": "Number of Events",
        "plot": ['hist'],
        "plot_keys": "content"
    },

    "repo_event_counts_issue": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Issue Events",
        "plot": ['hist']
    },

    "repo_event_counts_pull_request": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Pull Requests",
        "plot": ['hist']
    },

    "repo_event_counts_push": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos",
        "x_axis": "Number of Push Events",
        "plot": ['hist']
    },

    "content_event_distribution_daily": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "# Events",
        "plot": ['multi_time_series'],
        "plot_keys": "content"
    },

    "content_event_distribution_dayofweek": {
        "data_type": "dict_DataFrame",
        "x_axis": "Day of Week",
        "y_axis": "# Events",
        "plot": ['multi_time_series'],
        "plot_keys": "content"
    },

    "content_growth": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "# Events",
        "plot": ['time_series'],
        "plot_keys": "content"
    },
    #
    "repo_issue_to_push": {
        "data_type": "dict_DataFrame",
        "x_axis": "Number of Previous Events",
        "y_axis": "Issue Push Ratio",
        "plot": ['time_series'],
        "plot_keys": "content"
    },

    "content_liveliness_distribution": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos/Posts/Tweets",
        "x_axis": "Number of Forks/Comments/Replies",
        "plot": ['hist']
    },

    "repo_trustingness": {
        "data_type": "DataFrame",
        "x_axis": "Ground Truth",
        "y_axis": "Simulation",
        "plot": ['scatter']
    },

    "content_popularity_distribution": {
        "data_type": "DataFrame",
        "y_axis": "Number of Repos/Tweets",
        "x_axis": "Number of Watches/Rewtweets",
        "plot": ['hist']
    },

    "repo_user_continue_prop": {
        "data_type": "dict_DataFrame",
        "x_axis": "Number of Actions",
        "y_axis": "Probability of Continuing",
        "plot": ['time_series'],
        "plot_keys": "content"
    },
    #
    #
    # ### user

    "user_popularity": {
        "data_type": "DataFrame",
        "y_axis": "Number of Users",
        "x_axis": "Popularity of User's Repos/Tweets/Posts",
        "plot": ['hist']
    },

    "user_activity_distribution": {
        "data_type": "DataFrame",
        "x_axis": "User Activity",
        "y_axis": "Number of Users",
        "plot": ['hist']
    },

    "user_diffusion_delay": {
        "data_type": "Series",
        "x_axis": "Diffusion Delay (H)",
        "y_axis": "Number of Events",
        "plot": ['hist']
    },
    "user_activity_timeline": {
        "data_type": "dict_DataFrame",
        "x_axis": "Date",
        "y_axis": "Number of Events",
        "plot": ['time_series'],
        "plot_keys": "user"
    },

    "user_trustingness": {
        "data_type": "DataFrame",
        "x_axis": "Ground Truth",
        "y_axis": "Simulation",
        "plot": ['scatter']
    },

    "user_unique_content": {
        "data_type": "DataFrame",
        "x_axis": "Number of Unique Repos/Posts/Tweets",
        "y_axis": "Number of Users",
        "plot": ['hist']
    }
}

node_measurements = {
    'group_versus_total_volume_of_activity': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Fraction of Total Activity from Most Prolific Group',
        'plot_keys': 'informationID'
    },
    'node_unique_users': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Unique Users',
        'plot_keys': 'informationID'},
    'node_number_of_shares_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Number of Shares',
        'plot_keys': 'informationID'
    },
    'node_number_of_shares':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Node',
         'y_axis': 'Number of Shares',
         'plot_keys': 'informationID'},
    'node_lifetime_of_info':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Node',
         'y_axis': 'Lifetime',
         'plot_keys': 'informationID'},
    'node_speed':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Node',
         'y_axis': 'Speed',
         'plot_keys': 'informationID'},
    'node_unique_users_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Number of Unique Users',
        'plot_keys': 'informationID'
    },
    'node_top_info_shared': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Shares',
        'plot_keys': 'informationID'
    },
    'node_top_audience_reach': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Audience Size',
        'plot_keys': 'informationID'
    },
    'node_lifetime_of_threads': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Thread Lifetime',
        'y_axis': 'Number of Threads',
        'plot_keys': 'informationID'
    },
    'node_top_lifetime': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Lifetime',
        'plot_keys': 'informationID'
    },
    'node_distribution_of_lifetimes': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Lifetime',
        'y_axis': 'Number of Information Units'
    },
    'node_speed_of_info_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Information Speed',
        'plot_keys': 'informationID'
    },
    'node_distribution_of_speed': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Speed',
        'plot_keys': 'informationID'
    },
    'node_top_speed': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Speed',
        'plot_keys': 'informationID'
    },
    'node_distribution_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Number of Shares',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'informationID'
    },
    'node_distribution_of_users': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Users',
        'plot_keys': 'informationID'
    },
    'node_average_size_of_each_burst': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Average Size',
        'y_axis': 'Number of Bursts',
        'plot_keys': 'informationID'
    },
    'node_number_of_bursts': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'InformationID',
        'y_axis': 'Number of Bursts'
    },
    'node_time_between_bursts': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Time Between Bursts',
        'plot_keys': 'informationID'
    },
    'node_average_number_of_users_per_burst': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'InformationID',
        'y_axis': 'Average Number of users per burst',
    },
    'node_average_proportion_of_top_platform_per_burst': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'InformationID',
        'y_axis': 'Average Proportion of Top Platform per Burst'
    },
    'node_burstiness_of_burst_timing': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'InformationID',
        'y_axis': 'Burstiness'
    },
    'node_lifetime_of_each_burst': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'InformationID',
        'y_axis': 'Average Lifetime of Burst'
    },
    'node_new_users_per_bursts': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'New Users',
        'y_axis': 'Number of InformationIDs',
        'plot_keys': 'informationID'
    },
    'node_lifetime_of_bursts': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Lifetime',
        'y_axis': 'Number of InformationIDs',
        'plot_keys': 'informationID'
    },
    'population_distribution_of_average_burst_size': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Average Burst Size',
        'y_axis': 'Number of InformationIDs'
    },
    'population_distribution_of_average_number_of_users_per_burst': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Average Number of Users Per Burst',
        'y_axis': 'Number of InformationIDs'
    },
    'population_distribution_of_burst_lifetime': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Average Burst Lifetime',
        'y_axis': 'Number of InformationIDs'
    },
    'population_distribution_of_burst_platform_proportion': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Burst Top Platform Proportion',
        'y_axis': 'Number of InformationIDs'
    },
    'population_distribution_of_new_users_per_burst': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'New Users Per Burst',
        'y_axis': 'Number of InformationIDs'
    },
    'population_distribution_of_number_of_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Number of Bursts',
        'y_axis': 'Number of InformationIDs'
    },
    'node_top_platform': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Average Proportion',
        'plot_keys': 'informationID'
    },
    'node_overlapping_users': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'informationID'
    },
    'node_order_of_spread': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'RBO Score',
        'plot_keys': 'informationID'
    },
    'node_time_delta': {
        'data_type': 'dict',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Time',
        'plot_keys': 'informationID'
    },
    'node_audience_size': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Size',
        'plot_keys': 'informationID'
    },
    'node_speed_of_spread': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Speed',
        'plot_keys': 'informationID'
    },
    'node_size_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Size',
        'plot_keys': 'informationID'
    },
    'node_temporal_correlation_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Shares',
        'plot_keys': 'informationID'
    },
    'node_temporal_correlation_of_audience': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Audience Size',
        'plot_keys': 'informationID'
    },
    'node_lifetime': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Lifetime',
        'plot_keys': 'informationID'
    },
    'node_correlation_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Shares',
        'plot_keys': 'informationID'
    },
    'node_correlation_of_audience_sizes': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Audience Size',
        'plot_keys': 'informationID'
    },
    'node_correlation_of_lifetimes': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Lifetimes',
        'plot_keys': 'informationID'
    },
    'node_correlation_of_speeds': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Correlation of Speeds',
        'plot_keys': 'informationID'
    }
}
community_measurements = {

    'cascade_participation_gini': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'Cascade',
        'y_axis': 'Cascade Participation Gini Coefficient'
    },
    'cascade_participation_palma': {
        'data_type': 'dict',
        'plot': ['bar'],
        'x_axis': 'Cascade',
        'y_axis': 'Cascade Participation Palma Ratio'
    },
    'cascade_breadth_by_depth': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Depth',
        'y_axis': 'Breadth',
        'plot_keys': 'cascade'},

    'cascade_breadth_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Breadth',
         'plot_keys': 'cascade'},

    'cascade_max_depth_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Depth',
         'plot_keys': 'cascade'},

    'cascade_new_user_ratio_by_depth':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Depth',
         'y_axis': 'New User Ratio',
         'plot_keys': 'cascade'},

    'cascade_new_user_ratio_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'New User Ratio',
         'plot_keys': 'cascade'},

    'cascade_size_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Size',
         'plot_keys': 'cascade'},

    'cascade_structural_virality_over_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Structural Virality',
         'plot_keys': 'cascade'},

    'cascade_uniq_users_by_depth':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Depth',
         'y_axis': 'Unique Users',
         'plot_keys': 'cascade'},

    'cascade_uniq_users_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Unique Users',
         'plot_keys': 'cascade'},

    'community_cascade_lifetime_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Lifetime',
         'y_axis': 'Number of Cascades',
         'plot_keys': 'community'},

    'community_cascade_lifetime_timeseries':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Lifetime',
         'plot_keys': 'community'},
    'community_cascade_participation_palma':
        {
            'data_type': 'dict',
            'plot': ['bar'],
            'x_axis': 'Community',
            'y_axis': 'Palma',
        },
    'community_cascade_initialization_palma':
        {
            'data_type': 'dict',
            'plot': ['bar'],
            'x_axis': 'Community',
            'y_axis': 'Palma',
        },
    'community_cascade_participation_gini':
        {
            'data_type': 'dict',
            'plot': ['bar'],
            'x_axis': 'Community',
            'y_axis': 'Gini',
        },
    'community_cascade_initialization_gini':
        {
            'data_type': 'dict',
            'plot': ['bar'],
            'x_axis': 'Community',
            'y_axis': 'Gini',
        },

    'community_cascade_size_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Size',
         'y_axis': 'Number of Cascades',
         'plot_keys': 'community'},

    'community_cascade_size_timeseries':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Time',
         'y_axis': 'Cascade Size',
         'plot_keys': 'community'},

    'community_max_breadth_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Breadth',
         'y_axis': 'Number of Cascades',
         'plot_keys': 'community'},

    'community_max_depth_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Depth',
         'y_axis': 'Number of Cascades',
         'plot_keys': 'community'},

    'community_new_user_ratio_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'New User Ratio',
         'plot_keys': 'community'},

    'community_structural_virality_distribution':
        {'data_type': 'dict_DataFrame',
         'plot': ['hist'],
         'x_axis': 'Structural Virality',
         'y_axis': 'Number of Cascade',
         'plot_keys': 'community'},
    'community_unique_users':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Community',
         'y_axis': 'Number of Unique Users',
         'plot_keys': 'community'},
    'community_unique_users_by_time':
        {'data_type': 'dict_DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Time',
         'y_axis': 'Unique Users',
         'plot_keys': 'community'},
    'community_number_of_shares_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Number of Shares Per Information Unit',
        "plot_keys":"community"
    },
    'community_number_of_shares':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Community',
         'y_axis': 'Number of Shares',
         'plot_keys': 'community'},
    'community_lifetime_of_info':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Lifetime',
         'y_axis': 'Number of Information Units',
         'plot_keys': 'community'},
    'community_speed':
        {'data_type': 'dict',
         'plot': ['hist'],
         'x_axis': 'Community',
         'y_axis': 'Speed',
         'plot_keys': 'community'},
    'community_unique_users_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Audience Size Per Information ID',
        'plot_keys': 'community'
    },
    'community_top_info_shared': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Number of Shares',
        'plot_keys': 'community'
    },
    'community_top_audience_reach': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Audience Size',
        'plot_keys': 'community'
    },
    'community_lifetime_of_threads': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Mean Thread Lifetime',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_top_lifetime': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Lifetime',
        'plot_keys': 'community'
    },
    'community_distribution_of_lifetimes': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Lifetime',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_speed_of_info_over_time': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Information Speed Per Information Unit',
        'plot_keys': 'community'
    },
    'community_distribution_of_speed': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Speed',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_top_speed': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Speed',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_distribution_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Number of Shares',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_distribution_of_users': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Audience Size',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_number_of_bursts': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Number of Bursts',
        'plot_keys': 'community'
    },
    'community_time_between_bursts': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Time Between Bursts',
        'plot_keys': 'community'
    },
    'community_average_number_of_users_per_bursts': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Number of Users',
        'plot_keys': 'community'
    },
    'community_burstiness_of_burst_timing': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Burstiness',
        'plot_keys': 'community'
    },
    'community_new_users_per_bursts': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'New Users',
        'plot_keys': 'community'
    },
    'community_lifetime_of_bursts': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Lifetime',
        'plot_keys': 'community'
    },
    'community_top_platform': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Average Proportion',
        'plot_keys': 'community'
    },
    'community_overlapping_users': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'community'
    },
    'community_order_of_spread': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'RBO Score',
        'plot_keys': 'community'
    },
    'community_time_delta': {
        'data_type': 'dict_DataFrame',
        'plot': ['multi_hist'],
        'x_axis': 'Time Elapsed',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_audience_size': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Size',
        'plot_keys': 'community'
    },
    'community_speed_of_spread': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Speed',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_size_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Size',
        'plot_keys': 'community'
    },
    'community_temporal_correlation_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['multi_hist'],
        'x_axis': 'Correlation of Number of Shares',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_temporal_correlation_of_audience': {
        'data_type': 'dict_DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Correlation of Audience Size',
        'y_axis': 'Number of Information Units',
        'plot_keys': 'community'
    },
    'community_lifetime': {
        'data_type': 'dict_DataFrame',
        'plot': ['hist'],
        'x_axis': 'Community',
        'y_axis': 'Lifetime',
        'plot_keys': 'community'
    },
    'community_correlation_of_shares': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'community'
    },
    'community_correlation_of_audience_sizes': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'community'
    },
    'community_correlation_of_lifetimes': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'community'
    },
    'community_correlation_of_speeds': {
        'data_type': 'dict_DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform',
        'plot_keys': 'community'
    }
}



population_measurements = {
    'group_size_distribution':
        {'data_type': 'list',
         'plot': ['hist'],
         'x_axis': 'Number of Users in Group',
         'y_axis': 'Number of Groups'},

    'seed_post_versus_response_actions_ratio':
        {'data_type': 'list',
         'plot': ['hist'],
         'x_axis': 'Seed post to total actions ratio',
         'y_axis': 'Number of Groups'},

    'distribution_of_content_discussion_over_groups':
        {'data_type': 'list',
         'plot': ['hist'],
         'x_axis': 'Gini Coefficient for Content Discussion',
         'y_axis': 'Number of Groups'},

    'population_cascade_lifetime_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Cascade Lifetime',
         'y_axis': 'Number of Cascades'},

    'population_cascade_lifetime_timeseries':
        {'data_type': 'DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Lifetime'},

    'population_cascade_size_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Size',
         'y_axis': 'Number of Cascades'},

    'population_cascade_size_timeseries':
        {'data_type': 'DataFrame',
         'plot': ['time_series'],
         'x_axis': 'Date',
         'y_axis': 'Cascade Size'},

    'population_max_breadth_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Breadth',
         'y_axis': 'Number of Cascades'},

    'population_max_depth_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Max Depth',
         'y_axis': 'Number of Cascades'},

    'population_structural_virality_distribution':
        {'data_type': 'DataFrame',
         'plot': ['hist'],
         'x_axis': 'Structural Virality',
         'y_axis': 'Number of Cascade'},

    'population_degree_distribution': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Degree',
        'y_axis': 'Number of Nodes'
    },
    'population_number_of_shares_over_time': {
        'data_type': 'DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Number of Shares Per Information Unit'
    },
    'population_unique_users_over_time': {
        'data_type': 'DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Audience per Information Unit'
    },
    'population_top_info_shared': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Shares'
    },
    'population_top_audience_reach': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Audience Size'
    },
    'population_lifetime_of_threads': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Mean Thread Lifetime',
        'y_axis': 'Number of Information Units'
    },
    'population_top_lifetime': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Lifetime'
    },
    'population_distribution_of_lifetimes': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Lifetime',
        'y_axis': 'Number of Information Units'
    },
    'population_speed_of_info_over_time': {
        'data_type': 'DataFrame',
        'plot': ['time_series'],
        'x_axis': 'Time',
        'y_axis': 'Average Information Speed Per Information Unit'
    },
    'population_distribution_of_speed': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Speed',
        'y_axis': 'Number of Information Units'
    },
    'population_top_speed': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Speed'
    },
    'population_distribution_of_shares': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Number of Shares',
        'y_axis': 'Number of Information Units'
    },
    'population_distribution_of_users': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Audience Size',
        'y_axis': 'Number of Units of Information'
    },
    'population_average_size_of_each_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Average Size'
    },
    'population_number_of_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Bursts'
    },
    'population_time_between_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Time Between Bursts'
    },
    'population_average_number_of_users_per_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Number of Users'
    },
    'population_burstiness_of_burst_timing': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Burstiness'
    },
    'population_new_users_per_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'New Users'
    },
    'population_lifetime_of_bursts': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Lifetime'
    },
    'population_top_platform': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Average Proportion'
    },
    'population_overlapping_users': {
        'data_type': 'DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform'
    },
    'population_order_of_spread': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'RBO Score'
    },
    'population_time_delta': {
        'data_type': 'DataFrame',
        'plot': ['multi_hist'],
        'x_axis': 'Time Elapsed',
        'y_axis': 'Number of Information Units'
    },
    'population_audience_size': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Size'
    },
    'population_speed_of_spread': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Speed'
    },
    'population_size_of_shares': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Size'
    },
    'population_temporal_correlation_of_shares': {
        'data_type': 'DataFrame',
        'plot': ['multi_hist'],
        'x_axis': 'Correlation of Shares',
        'y_axis': 'Number of Information Units'
    },
    'population_temporal_correlation_of_audience': {
        'data_type': 'DataFrame',
        'plot': ['multi_hist'],
        'x_axis': 'Correlation of Audience Size',
        'y_axis': 'Number of Information Units'
    },
    'population_lifetime': {
        'data_type': 'DataFrame',
        'plot': ['hist'],
        'x_axis': 'Node',
        'y_axis': 'Lifetime'
    },
    'population_correlation_of_shares': {
        'data_type': 'DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform'
    },
    'population_correlation_of_audience_sizes': {
        'data_type': 'DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Correlation of Audience Size'
    },
    'population_correlation_of_lifetimes': {
        'data_type': 'DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform'
    },
    'population_correlation_of_speeds': {
        'data_type': 'DataFrame',
        'plot': ['heatmap'],
        'x_axis': 'Platform',
        'y_axis': 'Platform'
    }
}

measurement_plot_params.update(node_measurements)
measurement_plot_params.update(community_measurements)
measurement_plot_params.update(population_measurements)
