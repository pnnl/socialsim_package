from matplotlib import pyplot as plt
import seaborn as sns

import re
import pickle
import pandas as pd
import numpy as np

def load_metrics(fns, narratives=[]):

    dfs = []
    model_counts = {}
    for fn in fns:
        
        with open(fn,'rb') as f:
            res = pickle.load(f)

        model = res['model_identifier']
        if model not in model_counts.keys():
            model_counts[model] = 0
        else:
            model_counts[model] += 1
        
        for platform in res['metrics'].keys():
            for meas_type in res['metrics'][platform].keys():
                metrics = res['metrics'][platform][meas_type]['node']
                for m in metrics.keys():
                    if metrics[m] is None: continue
                    metric_results = metrics[m][list(metrics[m].keys())[0]]
                
                    df = pd.Series(metric_results)
                    df = pd.DataFrame(df).reset_index()
                    df.columns = ['narrative','value']
                    df = df[df['narrative'].isin(narratives)]
                    df['platform'] = platform
                    df['measurement'] = m
                    df['metric'] = list(metrics[m].keys())[0]
                    df['sample'] = model_counts[model]
                    df['model'] = model
                    
                    dfs.append(df)
                
    df = pd.concat(dfs).reset_index(drop=True)

    return(df)

def load_measurements(fns,narratives=[],meas_list_scalar=[],meas_list_temporal=[]):


    gt_dfs = []
    sim_dfs = []
    model_counts = {}
    for fn in fns:
    
    
        with open(fn,'rb') as f:
            res = pickle.load(f)


        model = res['model_identifier']
        if model not in model_counts.keys():
            model_counts[model] = 0
        else:
            model_counts[model] += 1

            
        gt = res['ground_truth_results']
        sim = res['simulation_results']

        for platform in ['twitter','youtube']:

            if platform not in gt.keys():
                continue

            if 'multi_platform' not in gt[platform].keys():
                continue

            for meas in meas_list_scalar:
                if model_counts[model] == 0:
                    df = pd.DataFrame.from_dict(gt[platform]['multi_platform']['node'][meas],
                                                orient='index').reset_index()
                    df.columns = ['informationID','value']
                    df = df[df['informationID'].isin(narratives)]
                    df['meas'] = meas
                    df['platform'] = platform

                    gt_dfs.append(df)

                df = pd.DataFrame.from_dict(sim[platform]['multi_platform']['node'][meas],
                                            orient='index').reset_index()
                df.columns = ['informationID','value']
                df = df[df['informationID'].isin(narratives)]

                df['meas'] = meas
                df['sample'] = model_counts[model]
                df['platform'] = platform
                df['model'] = model
                
                sim_dfs.append(df)

            for meas in meas_list_temporal:

                if model_counts[model] == 0:
                    for key,val in gt[platform]['multi_platform']['node'][meas].items():
                        val['informationID'] = key
                        val['meas'] = meas
                        val['platform'] = platform

                        if key in narratives:
                            gt_dfs.append(val)

                for key,val in sim[platform]['multi_platform']['node'][meas].items():
                    val['informationID'] = key
                    val['meas'] = meas
                    val['sample'] = model_counts[model]
                    val['platform'] = platform
                    val['model'] = model

                    if key in narratives:
                        sim_dfs.append(val)


    sim_df = pd.concat(sim_dfs)
    gt_df = pd.concat(gt_dfs)

    return(gt_df,sim_df)
        
    


def strip_plot(grouped,platforms=[],meas_list=[],log=True,narrative='',
               save_plots=False,save_path='./'):

    measurements = grouped[['measurement','metric']].drop_duplicates()
    measurements = measurements[measurements['measurement'].isin(meas_list)]
    meas_list = measurements['measurement'].values
    metric_list = measurements['metric'].values

    model_names = list(grouped['model'].unique())
    
    pal = sns.color_palette()
        
    for platform in platforms:
        for m,meas in enumerate(meas_list):
            sns.set_context("talk")
            fig, ax = plt.subplots(figsize=(10,8))

            n_info_ids = grouped['narrative'].nunique()
            
            
            grouped['jittered'] = grouped['model'].apply(lambda x: model_names.index(x)) + np.random.uniform(-0.15,
                                                                                                             0.15,
                                                                                                             len(grouped))


            subset = grouped[ (grouped['measurement'] == meas) & (grouped['platform'] == platform)]
            count = 0
            for key,grp in subset.groupby('narrative'):

                if key != narrative and narrative != '':
                    alpha = 0.5
                else:
                    alpha = 1.0

                grp.plot(x='jittered',y='mean',kind = 'scatter', yerr='std',ax=ax,
                         s=50,
                         legend=False,
                         color=pal[0],
                         gid=key,
                         picker=20,
                         alpha = alpha)

                    
                if key == narrative and narrative != '':

                    color = pal[0]
                    
                    grp.plot(x='jittered',y='mean',kind = 'scatter', yerr='std',ax=ax,
                             s=50,
                             color='k')
                    grp.plot(x='jittered',y='mean', yerr='std',ax=ax,
                             label=key,
                             color='k',linestyle='--')

                    
                count += 1

            plt.ylabel(metric_list[m])
            plt.xlabel('Model')
            plt.xticks(range(len(model_names)),model_names)

            if subset['mean'].max() / (subset['mean'][subset['mean'] > 0]).min() > 100 and log:
                plt.yscale('log')

            plt.title(f'{meas} - {platform}\nError By Model and Info ID')
            if save_plots:
                if narrative == '':
                    plt.savefig(f'{save_path}/{meas}_{platform}_stripplot.png')
                else:
                    plt.savefig(f'{save_path}/{meas}_{platform}_{narrative}_stripplot.png')
                    

def calculate_ccdf(df, min_errors=[],max_errors=[],value_col='value'):


    min_error = df[value_col].min()
    nonzero = df[ (df[value_col] != 0) ][value_col].min()
    max_error = df[value_col].max()
    max_errors.append(max_error)
    min_errors.append(nonzero)
    error_pts = np.unique(np.sort(np.concatenate([max_error*np.arange(1000)/999.0,df[value_col].unique()])))
    ccdfs = []
    for i in range(10):
        sample = df.copy()
        sample = sample[(sample['sample'] == i)]
        
        sample = sample.sort_values(value_col)
        sample['ccdf'] = (sample[value_col] / sample[value_col].sum()).cumsum()

        sample = sample.drop_duplicates(subset=value_col,keep='last')
                    
        sample = sample.set_index(value_col)


        sample = sample.reindex(error_pts).interpolate().fillna(0)

        sample = sample.reset_index()

        sample['sub'] = i

        ccdfs.append(sample)


    ccdfs = pd.concat(ccdfs)
    ccdfs = ccdfs.groupby([value_col])['ccdf'].agg([np.mean,np.min,np.max]).reset_index()

    return(ccdfs,min_errors,max_errors)


def ccdf_plot(df,platform,meas,log=True,
              save_plots=False,save_path='./'):

    measurements = df[['measurement','metric']].drop_duplicates()
    measurements = measurements[measurements['measurement'] == meas]
    metric = measurements['metric'].values[0]

    model_names = list(df['model'].unique())
    
    
    fig, ax = plt.subplots(figsize=(10,8))

    pal = sns.color_palette()
    
    count = 0
    max_errors = []
    min_errors = []

    for model in df['model'].unique():

        idx = (df['model'] == model) & (df['measurement'] == meas) & (df['platform'] == platform)
        
        ccdfs,min_errors,max_errors = calculate_ccdf(df[idx],min_errors,max_errors)        
        
        if len(ccdfs) == 0:
            continue

        ccdfs.plot(x='value',y='mean',label=str(model), c = pal[count],ax=ax)
        ax.fill_between(ccdfs['value'].values,ccdfs['amin'].values,ccdfs['amax'].values,color=pal[count],alpha=0.3)

        count += 1
        
    plt.xlim(min(min_errors),max(max_errors)) 
    if max(max_errors)/min(min_errors) > 100 and log: 
        plt.xscale('log')
    plt.xlabel(metric)
    plt.ylabel('Proportion of Information IDs\nWith Error Less than Or Equal')
    plt.title(f'{platform} {meas}\nCCDF of {metric} Across Info IDs')
    if save_plots:
        plt.savefig(f'{save_path}/{meas}_{platform}_ccdf.png')


def ccdf_plots(df,platforms=[],meas_list=[],log=True,
               save_plots=False,save_path='./'):

    for platform in platforms:
        for m,meas in enumerate(meas_list):

            ccdf_plot(df,platform,meas,log=log,
                      save_plots=save_plots,save_path=save_path)


def time_series_plot(df,platform,meas,narrative,log=False,
                     save_plots=False,save_path='./'):

    pal = sns.color_palette()
    
    fig, ax = plt.subplots(figsize=(10,8))

    grp = df[ (df['meas'] == meas) & (df['platform'] == platform) & (df['informationID'] == narrative) ]

    models = grp['model'].unique()
    
    grp[grp['model'] == models[0]].plot(x='nodeTime',y='value',color='k',label='Ground Truth',ax=ax)

    count = 0
    for g2, grp2 in grp.groupby('model'):    
  
        grp2.plot(x='nodeTime',y='mean',label=g2,
                  ax=ax,color=pal[count])
        plt.fill_between(grp2['nodeTime'].values,(grp2['mean'] - grp2['std']).values,
                         (grp2['mean'] + grp2['std']).values,color=pal[count],alpha=0.3)
                

        count += 1

    if log:
        plt.yscale("log")
        
    plt.ylabel(meas)
    plt.title(f'{narrative} - {meas} - {platform}')

    if save_plots:
        plt.savefig(f'{save_path}/{meas}_{platform}_{narrative}_timeseries.png')

    
def scatter_plot(df, platforms=[],meas_list=[], log=True,
                 save_plots=False,save_dir='./'):

    sns.set_context("talk",font_scale=1.2)
    pal = sns.color_palette()
    for platform in platforms:
        for meas in meas_list:
            subset = df[(df['meas'] == meas) & (df['platform'] == platform)]

            fig, ax = plt.subplots(figsize=(10,8))

            count = 0
            for g,grp in subset.groupby('model'):
                grp.plot(x='value',y='mean',kind='scatter',yerr='std',label=g,ax=ax,color=pal[count])
                count += 1

            max_val = max([subset['value'].max(),subset['mean'].max()])
            min_val = min([subset['value'].min(),subset['mean'].min()])

            plt.plot([0.9*min_val,1.1*max_val],[0.9*min_val,1.1*max_val],linestyle='--',color='k')

            plt.ylim(0.9*min_val,1.1*max_val)
            plt.xlim(0.9*min_val,1.1*max_val)

            if log:
                plt.xscale("log")
                plt.yscale("log")

            plt.xlabel('Ground True Value')
            plt.ylabel('Model Value')
            plt.title(f'{meas} - {platform}')


            if save_plots:
                plt.savefig(f'{save_dir}/{meas}_{platform}_sim_vs_gt_scatter.png')

            
def pairwise_scatterplots(df,platforms=[],meas_list=[],log=True,
                          save_plots=False,save_dir='./'):

    measurements = df[['measurement','metric']].drop_duplicates()
    measurements = measurements[measurements['measurement'].isin(meas_list)]
    metric_list = measurements['metric'].values
    meas_list = list(measurements['measurement'].values)

    
    sns.set_context("talk",font_scale=1.2)
    pal = sns.color_palette()
    for platform in platforms:
        for meas in meas_list:
            metric = metric_list[meas_list.index(meas)]
            
            subset = df[(df['measurement'] == meas) & (df['platform'] == platform)]

            n_models = subset['model'].nunique()

            max_val = subset['mean'].max()
            min_val = subset['mean'].min()
            
            fig, axes = plt.subplots(figsize=(12,12),nrows=n_models,ncols=n_models,
                                    sharex=True,sharey=True)

            count = 0
            for m1,model1 in enumerate(subset['model'].unique()):
                for m2,model2 in enumerate(subset['model'].unique()):

                    
                    if model1 == model2:

                        if m2 == 0:
                            axes[m1][m2].set_ylabel(model1)
                        else:
                            axes[m1][m2].set_ylabel('')

                        if m1 == subset['model'].nunique() - 1:
                            axes[m1][m2].set_xlabel(model2)
                        else:
                            axes[m1][m2].set_xlabel('')

                        
                        continue


                    subset1 = subset[subset['model'] == model1]
                    subset2 = subset[subset['model'] == model2]

                    merged = subset1.merge(subset2,on=['narrative','platform','measurement','metric'],suffixes=('_1','_2'))

            
                    merged.plot(x='mean_1',y='mean_2',kind='scatter',yerr='std_1',xerr='std_2',
                             ax=axes[m1][m2],color=pal[0])
                    count += 1

                    if m2 == 0:
                        axes[m1][m2].set_ylabel(model1)
                    else:
                        axes[m1][m2].set_ylabel('')

                    if m1 == subset['model'].nunique() - 1:
                        axes[m1][m2].set_xlabel(model2)
                    else:
                        axes[m1][m2].set_xlabel('')
 
                    axes[m1][m2].plot([0.9*min_val,1.1*max_val],[0.9*min_val,1.1*max_val],linestyle='--',color='k')

            subset = subset[ (subset['mean'] > 0) ]
            max_val = subset['mean'].max()
            min_val = subset['mean'].min()
            
            plt.ylim(0.9*min_val,1.1*max_val)
            plt.xlim(0.9*min_val,1.1*max_val)

            if max_val/min_val > 100 and log:
                plt.xscale("log")
                plt.yscale("log")

 

            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

 
            plt.title(f'{meas} - {metric} - {platform}')


            if save_plots:
                plt.savefig(f'{save_dir}/{meas}_{platform}_pairwise_scatter.png')
