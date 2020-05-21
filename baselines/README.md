# README

The CP4 baseline generation script can be used by modifying the following parameters in the main function:
* path - location of extracted JSON files
* n_runs - the number of baseline files to generate for each simulation period
* simulation_periods - a list of the start and end dates of the time period you would like to generate baselines for

To use the script as is, the data files should have the following naming format:

venezuela_v2_extracted_\[platform\]\_\[start_date\]\_\[end_date\].json

For example:

venezuela_v2_extracted_twitter_2019-01-18_2019-01-25.json

For each simulation period, the script will try to find the files that cover the period just preceding the simulation period and of the same length as the simulation period to use as historical data for sampling.  For example, if you want to generate a baseline for the dates 2019-02-01 through 2019-02-08, the script will look for any file that starts before 2019-02-01 but that do not start earlier than 2019-01-25 (1 week prior). If you have weekly data files, you should split those files at dates that are consistent with your desired simulation period. For example, if you have a data file spanning from 2019-01-27 to 2019-02-04, then the data selection for the baseline *will not work correctly* for the specified simulation period. To avoid this issue, you could also split your data into daily files rather than weekly. In this example, the script would then load all the files spanning from 2019-01-25 through 2019-01-31 to use as historical data for sampling. This would allow you to more flexibly select simulation periods of different lengths.
