from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from scipy.stats import iqr
from scipy.stats import spearmanr
from scipy.stats import wasserstein_distance

from scipy.spatial.distance import euclidean
from scipy.optimize import minimize_scalar
from sklearn.metrics        import r2_score
from bisect import bisect

import fastdtw as fdtw
import pandas  as pd
import numpy   as np

import traceback

from .record import RecordKeeper

from .visualizations import generate_plot

import re


def list_metrics():
    empty_metrics_object = Metrics(None, None, None)
    count = 0
    for f in dir(empty_metrics_object):
        if not f.startswith('_'):
            func = getattr(empty_metrics_object, f)
            if callable(func):
                doc_string = func.__doc__
                if not doc_string is None and 'Metric:' in doc_string:
                    #print(doc_string)
                    desc = re.search('Description\:([\s\S]+?)Input', doc_string).groups()[0].strip()
                    desc = desc.replace('\n', ' ').replace('\t', ' ')  # remove new lines, tabs, double spaces
                    ## remove duplicate spaces:
                    while "  " in desc:
                        desc = desc.replace('  ',' ')
                    print('{}) {}: {}\n'.format(count + 1, f, desc))
                    count += 1

class Metrics:
    def __init__(self, ground_truth, simulation, configuration,
        log_file='metrics_log.txt', plot=False):
        """
        Description:

        Input:
            :ground_truth_measurements:
            :simulation_measurements:
            :configuration:

        Output:
            None

        """
        self.plot = plot

        self.ground_truth  = ground_truth
        self.simulation    = simulation
        self.configuration = configuration

        self.record_keeper = RecordKeeper('metrics_log.txt')

        pass


    def run(self, measurement_subset=None, verbose=False, plot_dir="./new_plots"):
        """
        Description: This runs all measurement outputs through the metrics
            specified in the configuration json.

            Note to reader:
                p for platform
                t for measurement type
                s for scale
                m for measurement

        Input:
            :measurement_subset: (list) The measurements to run metrics on. If
                None then metrics are run on all measurements.

        Output:
            :results: (dict)
            :logs: (dict)

        """
        results = {}
        logs    = {}

        for p in self.configuration.keys():
            p_ground_truth  = self.ground_truth[p]
            p_simulation    = self.simulation[p]
            p_configuration = self.configuration[p]

            p_results = {}
            p_logs    = {}

            for t in p_configuration:
                t_ground_truth  = p_ground_truth[t]
                t_simulation    = p_simulation[t]
                t_configuration = p_configuration[t]

                t_results = {}
                t_logs    = {}

                for s in t_configuration.keys():

                    if t_ground_truth is None:
                        s_ground_truth = t_ground_truth
                    else:
                        s_ground_truth = t_ground_truth[s]

                    if t_simulation is None:
                        s_simulation = t_simulation
                    else:
                        s_simulation = t_simulation[s]

                    s_configuration = t_configuration[s]

                    s_results = {}
                    s_logs    = {}

                    for m in s_configuration.keys():

                        if s_ground_truth is None:
                            ground_truth = s_ground_truth
                        else:
                            ground_truth = s_ground_truth[m]

                        if s_simulation is None:
                            simulation = s_simulation
                        else:
                            simulation = s_simulation[m]

                        configuration = s_configuration[m]

                        result, log = self._evaluate_metrics(ground_truth,
                                simulation, configuration, verbose, p,
                                t, s, m, plot_dir)

                        s_results.update({m:result})
                        s_logs.update({m:log})

                    t_results.update({s:s_results})
                    t_logs.update({s:s_logs})

                p_results.update({t:t_results})
                p_logs.update({t:t_logs})

            results.update({p:p_results})
            logs.update({p:p_logs})

        return results, logs


    def _evaluate_metrics(self, ground_truth, simulation, configuration,
        verbose, p, t, s, m, plot_dir):
        """
        Description: Evaluate metrics on a single measurement.

        Input:
            :ground_truth:
            :simulation:
            :configuration:

        Output:
        """
        log = {}
        all_results = {}

        if ground_truth is None:
            log.update({'status' : 'failure'})
            result = None

            return result, log

        if simulation is None:
            log.update({'status' : 'failure'})
            result = None

            return result, log

        if self.plot:
            generate_plot(
                simulation=simulation,
                ground_truth=ground_truth,
                measurement_name=m,
                plot_dir=plot_dir
                )

        # Loop over metrics
        for metric in configuration['metrics'].keys():
            # Get metric name and arguments
            metric_name = configuration['metrics'][metric]['metric']

            if verbose:
                message = 'SOCIALSIM METRICS      | Running '
                message = message + p+' '+t+' '+s+' '+m+' '
                message = message + metric_name
                message = message + '... '
                print(message, end='', flush=True)

            if 'metric_args' in configuration['metrics'][metric].keys():
                metric_args = configuration['metrics'][metric]['metric_args']
            else:
                metric_args = {}

            try:
                metric_function = getattr(self, metric_name)
            except Exception as error:
                result = metric_name+' was not found.'
                log.update({'status' : 'failure'})
                log.update({'error'  : error})

                if verbose:
                    print('')
                    print('-'*80)
                    trace = traceback.format_exc()
                    print(trace)

                return result, log

            self.record_keeper.tic(1)

            try:
                if s in ['node', 'community']:
                    result = {}
                    for a in ground_truth.keys():
                        if a in simulation.keys():
                            try:
                                sub_result = metric_function(ground_truth[a],
                                                             simulation[a], **metric_args)
                                result.update({a:sub_result})
                            except:
                                result.update({a:None})
                else:
                    result = metric_function(ground_truth, simulation,
                        **metric_args)
            except Exception as error:
                result = metric_name+' failed to run.'

                if verbose:
                    print('')
                    print('-'*80)
                    trace = traceback.format_exc()
                    print(trace)

            delta_time = self.record_keeper.toc(1)
            log.update({'run_time': delta_time})

            if verbose:
                message = 'Done. ({0} seconds.)'.format(delta_time)
                print(message, flush=True)

            all_results.update({metric:result})

        return all_results, log


    def check_data_types(self, ground_truth, simulation):
        """
        Convert ground truth and simulation measurements to arrays if they are
            DataFrames

        Inputs:
            :ground_truth: (unknown) Output of measurements code.
            :simulation: (unknown) Output of measurements code.
        Outputs:
            :ground_truth: (np.array) Standardized ground truth measurements
                data.
            :simulation: (np.array) Standardized simulation measurements data.
        """

        if isinstance(ground_truth, pd.DataFrame):
            ground_truth = ground_truth['value']
        if isinstance(simulation, pd.DataFrame):
            simulation = simulation['value']

        ground_truth = np.array(ground_truth)
        simulation = np.array(simulation)

        return ground_truth, simulation


    def get_hist_bins(self, ground_truth, simulation, method='auto'):
        """
        Calculate bins for combined ground truth and simulation data sets to
        use consistent bins for distributional comparisons

        Inputs:
            ground_truth: Ground truth measurement
            simulation: Simulation measurement
            method: Method of bin calculation corresponding the np.histogram bin
                argument

        Outputs:
            :bins: ()

        """
        all_data = np.concatenate([ground_truth, simulation])

        _, bins = np.histogram(all_data, bins=method)

        return (bins)


    def join_dfs(self, ground_truth, simulation, join='inner', fill_value=0):

        """
        Join the simulation and ground truth data frames

        Inputs:
        ground_truth - Ground truth measurement data frame with measurement in the
            "value" column
        simulation - Simulation measurement data frame with measurement in the
            "value" column
        join - Join method (inner, outer, left, right)
        fill_value - Value for filling NAs or method for filling in NAs
            (e.g. "ffill" for forward fill)
        """

        on = [c for c in ground_truth.columns if c!='value']

        suffixes = ('_gt', '_sim')

        df = ground_truth.merge(simulation, on=on, suffixes=suffixes, how=join)
        df = df.sort_values([c for c in ground_truth.columns if c != 'value'])

        try:
            float(fill_value)
            df = df.fillna(fill_value)
        except ValueError:
            df = df.fillna(method=fill_value)

        return(df)


    def rbo_weight(self, d, p):
        # Weight given to the top d ranks for a given p
        sum1 = 0.0
        for i in range(1, d):
            sum1 += np.power(p, i) / float(i)

        wt = 1.0-np.power(p,(d-1))+(((1-p)/p)*d)*(np.log(1/(1-p))-sum1)

        return wt


    """
    The remaining functions are metrics used in comparing the output of the
    measurements code.
    """


    def absolute_difference(self, ground_truth, simulation):
        """
        Metric: absolute_difference

        Description: Absolute difference between ground truth simulation measurement. Meant for scalar valued measurements.

        Input:
        """

        if not ground_truth is None and not simulation is None:
            result = np.abs(float(simulation) - float(ground_truth))
        else:
            result = None

        return result


    def absolute_percentage_error(self, ground_truth, simulation):
        """
        Metric: absolute_percentage_error

        Description: Absolute percentage error between ground truth simulation measurement
        Meant for scalar valued measurements

        Input:

        """

        if ground_truth is None or ground_truth==0 or simulation is None:
            result =  None
        else:
            result = self.absolute_difference(ground_truth, simulation)
            result = 100.0 * result / np.abs(float(ground_truth))

        return result


    def kl_divergence(self, ground_truth, simulation, discrete=False):
        """
        Metric: kl_divergence

        Description: KL Divergence between the ground truth and simulation data
        Meant for distributional measurements

        Inputs:
        ground_truth: Ground truth measurement
        simulation: Simulation measurement
        discrete: Whether the distribution is over discrete values (e.g. days of the week) (True) or numeric values (False)

        """
        if simulation is None:
            return None

        # if data is numeric, compute histogram
        if not discrete:

            ground_truth, simulation = self.check_data_types(ground_truth, simulation)

            bins = self.get_hist_bins(ground_truth, simulation,method='doane')

            ground_truth = np.histogram(ground_truth, bins=bins)[0]
            simulation = np.histogram(simulation, bins=bins)[0]

        else:
            df = ground_truth.merge(simulation,
                                    on=[c for c in ground_truth.columns if c != 'value'],
                                    suffixes=('_gt', '_sim'),
                                    how='outer').fillna(0)

            ground_truth = df['value_gt'].values.astype(float)
            simulation = df['value_sim'].values.astype(float)
            ground_truth = ground_truth / ground_truth.sum()
            simulation = simulation / simulation.sum()

        if len(ground_truth) == len(simulation):
            return entropy(ground_truth, simulation)
        else:
            print('Two distributions must have same length')
            return None


    def kl_divergence_smoothed(self, ground_truth, simulation, alpha=0.01,
        discrete=False):
        """
        Metric: kl_divergence_smoothed

        Description: Smoothed version of the KL divergence which smooths the simulation output
        to prevent infinities in the KL divergence output

        Input:

        Additional input:
        alpha - smoothing parameter
        """
        # if data is numeric, compute histogram
        if not discrete:
            ground_truth, simulation = self.check_data_types(ground_truth, simulation)

            bins = self.get_hist_bins(ground_truth, simulation)

            ground_truth = np.histogram(ground_truth, bins=bins)[0]
            simulation = np.histogram(simulation, bins=bins)[0]
            smoothed_simulation = (1 - alpha) * simulation + alpha * (np.ones(simulation.shape))
        else:
            df = ground_truth.merge(simulation,
                                    on=[c for c in ground_truth.columns if c != 'value'],
                                    suffixes=('_gt', '_sim'),
                                    how='outer').fillna(0)

            ground_truth = df['value_gt'].values.astype(float)
            simulation = df['value_sim'].values.astype(float)
            ground_truth = ground_truth / ground_truth.sum()
            simulation = simulation / simulation.sum()

        if len(ground_truth) == len(simulation):
            return entropy(ground_truth, smoothed_simulation)
        else:
            print('Two distributions must have same length')
            return None


    def dtw(self, ground_truth, simulation, join='outer', fill_value = 0):
        """
        Dynamic Time Warping implemenation
        """
        df = self.join_dfs(ground_truth,simulation,join=join,fill_value=fill_value)

        try:
            ground_truth = df['value_gt'].values
            simulation = df['value_sim'].values
        except:
            ground_truth = np.array(ground_truth)
            simulation = np.array(simulation)

        if len(simulation) > 0:
            dist = fdtw.dtw(ground_truth.tolist(), simulation, dist=euclidean)[0]
        else:
            dist = None

        return dist


    def fast_dtw(self, ground_truth, simulation, join = 'outer', fill_value = 0):
        """
        Fast Dynamic Time Warping implemenation
        """
        # Can we change this try except to a if statement?

        df = self.join_dfs(ground_truth,simulation,join=join,fill_value=fill_value)

        try:
            ground_truth = ground_truth['value'].values
            simulation = simulation['value'].values
        except:
            ground_truth = np.array(ground_truth)
            simulation = np.array(simulation)

        if len(simulation) > 0:
            dist = fdtw.fastdtw(ground_truth, simulation, dist=euclidean)[0]
        else:
            dist = None

        return dist


    def js_divergence(self, ground_truth, simulation, discrete=False, base=2.0):
        """
        Metric: js_divergence

        Description:  Jensen-Shannon Divergence implemenation
        A symmetric variant on KL Divergence which also avoids infinite outputs


        Input:

        ground_truth - ground truth measurement
        simulation - simulation measurement
        base - the logarithmic base to use
        """
        if simulation is None or len(simulation) == 0 or ground_truth is None or len(ground_truth)==0:
            return None

        if not discrete:


            ground_truth, simulation = self.check_data_types(ground_truth, simulation)

            try:
                ground_truth = ground_truth[np.isfinite(ground_truth)]
                simulation = simulation[np.isfinite(simulation)]
            except TypeError:
                return None

            bins = self.get_hist_bins(ground_truth, simulation,method='doane')

            ground_truth = np.histogram(ground_truth, bins=bins)[0].astype(float)
            simulation = np.histogram(simulation, bins=bins)[0].astype(float)

        else:

            try:
                ground_truth = ground_truth[np.isfinite(ground_truth.value)]
                simulation = simulation[np.isfinite(simulation.value)]
            except TypeError:
                return None

            df = ground_truth.merge(simulation,
                                    on=[c for c in ground_truth.columns if c != 'value'],
                                    suffixes=('_gt', '_sim'),
                                    how='outer').fillna(0)

            ground_truth = df['value_gt'].values.astype(float)
            simulation = df['value_sim'].values.astype(float)

        ground_truth = ground_truth / ground_truth.sum()
        simulation = simulation / simulation.sum()

        if len(ground_truth) == len(simulation):
            m = 1. / 2 * (ground_truth + simulation)
            return entropy(ground_truth, m, base=base) / 2. + entropy(simulation, m, base=base) / 2.
        else:
            print('Two distributions must have same length')
            return None


    def rbo_score(self, ground_truth, simulation, p=-1):
        """
        Metric: rbo_score

        Description: Rank biased overlap (RBO) implementation
        (http://codalism.com/research/papers/wmz10_tois.pdf)
        A ranked list comparison metric which allows non-overlapping lists

        Input:

        ground_truth - ground truth data
        simulation - simulation data
        p - RBO parameter ranging from 0 to 1 that determines how much to
            overweight the the upper portion of the list
            p = 0 means only the first element is considered
            p = 1 means all ranks are weighted equally
        """

        if type(ground_truth) is list or type(ground_truth) is np.ndarray:
            pass
        else:
            if len(ground_truth.columns) == 2:

                entity = [c for c in ground_truth.columns if c != 'value'][0]

                ground_truth = ground_truth[entity].tolist()
                simulation = simulation[entity].tolist()
            else:
                ground_truth = ground_truth.index.tolist()
                simulation = simulation.index.tolist()

        if p == -1:
            func = lambda x: np.abs(self.rbo_weight(len(ground_truth), x) - 0.995)
            res = minimize_scalar(func,method='bounded',bounds=(0,1))
            p = res.x


        sl, ll = sorted([(len(ground_truth), ground_truth), (len(simulation), simulation)])
        s, S = sl
        _, L = ll
        if s == 0: return 0

        # Calculate the overlaps at ranks 1 through s
        # (the shorter of the two lists)

        x_d = {}
        rbo_score = 0.0

        for i in range(1, s + 1):
            x = L[:i]
            y = S[:i]

            x_d[i] = len(set(x).intersection(set(y)))

        for i in range(1,s+1):
            rbo_score += (float(x_d[i])/float(i)) * pow(p, (i-1))

        rbo_score = rbo_score * (1 - p)

        return rbo_score

    def earth_movers(self,ground_truth,simulation):
        """
        Metric: earth_movers

        Description: Earth movers distance

        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column
        simulation - simulation measurement (data frame) with measurement in the
            "value" column
        """

        result = wasserstein_distance(ground_truth['value'],simulation['value'])

        return result

    def rh_distance(self,ground_truth,simulation):
        """
        Metric: rh_distance

        Description: Relative Hausdorff distance between the ground
        truth and simulation data.
        Meant for degree sequence measurements

        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column; measurement must be integers
        simulation - simulation measurement (data frame) with measurement in the
            "value" column; measurement must be integers

        Citation:
        Aksoy, S.G., K.E. Nowak, S.J. Young,
        "A linear time algorithm and analysis of graph Relative Hausdorff
        distance", SIAM Journal on Mathematics of Data Science 1,
        no. 4 (2019): 647-666.
        """

        #Rename for notional convience, convert to lists for speed
        G=list(ground_truth['value'])
        F=list(simulation['value'])

        #If no edges, RH computed on closest nonzero degree sequence of [1]
        if max(G)==0:
            G=[1]
        if max(F)==0:
            F=[1]

        #convert degree sequence to CCDH sequence
        G.sort()
        F.sort()
        G=[len(G) - bisect(G, z) for z in range(0,max(G))]
        F=[len(F) - bisect(F, z) for z in range(0,max(F))]

        #Convert CCDH sequence to dictionary of CCDH values
        #keyed by degrees 1 to max degree.
        mDegF = len(F)
        degF = {i : F[i-1] for i in range(1, mDegF+1)}
        mDegG = len(G)
        degG = {i : G[i-1] for i in range(1, mDegG+1)}

        # Main subroutine: Return max of epsilon, delta, where delta is
        # min size box around (d,Fd) containing smooth path through G.
        def epsilon_box(d,Fd,G,mDegG,epsilon):
            right = d*(1+epsilon)
            r = int(np.floor(right))
            r_frac = right - r
            rightF = Fd*(1+epsilon)

            left = d*(1-epsilon)
            k = int(np.ceil(left))
            k_frac = k - left
            leftF = Fd*(1-epsilon)

            if ((leftF <= mDegG)
                and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
                and (leftF <= G.get(k,G[1])*(1-k_frac) + G.get(k-1,G[1])*k_frac)):
                return epsilon

            if ((mDegG < leftF) and (leftF <= mDegG+1)
                and (rightF >= G.get(r,0)*(1-r_frac) + G.get(r+1,0)*r_frac)
                and (leftF <= G[mDegG]*k_frac)):
                return epsilon

            # If here, G doesn't pass through current epsilon box,
            # must find necessary epsilon box size. If G pass below box
            if G.get(d,0) < Fd:
                while k > mDegG + 1 or G.get(k-1,G[1]) < Fd*float(k-1)/d:
                    k -= 1
                if k == mDegG + 1:
                    return 1 - float(G[mDegG]*(mDegG+1))/(Fd + G[mDegG]*d)
                else:
                    return 1 - float((1-k)*G.get(k,G[1])
                            + k*G.get(k-1,G[1]))/(Fd + d*(G.get(k-1,G[1])
                            - G.get(k,G[1])))
            # otherwise, G pass above box
            else:
                while G.get(r+1,0) > Fd*float(r+1)/d:
                    r += 1
                return float((1+r)*G.get(r,0)
                            - r*G.get(r+1,0))/(d*(G.get(r,0)
                            - G.get(r+1,0)) + Fd) - 1

        #Compute RH distance
        checkF = set([1,mDegF] + [j for j in range(2,mDegF)
                    if not ((degF[j] == degF[j-1]) and (degF[j] == degF[j+1]))])
        checkG = set([1,mDegG] + [j for j in range(2,mDegG)
                    if not ((degG[j] == degG[j-1]) and (degG[j] == degG[j+1]))])
        epsilon = 0.0
        for d in range(1,max(mDegG,mDegF)+1):
            if d in checkF:
                epsilon_prime = epsilon_box(d,degF[d],degG,mDegG,epsilon)
                if epsilon_prime > epsilon:
                    epsilon = epsilon_prime
            if d in checkG:
                epsilon_prime = epsilon_box(d,degG[d],degF,mDegF,epsilon)
                if epsilon_prime > epsilon:
                    epsilon = epsilon_prime
        return epsilon


    def rmse(self, ground_truth, simulation, join='inner', fill_value=0,
        relative=False, cumulative=False, normed=False):
        """
        Metric: rmse

        Description: Root mean squared error

        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column
        simulation - simulation measurement (data frame) with measurement in the
            "value" column
        join - type of join to perform between ground truth and simulation
        fill_value - fill value for non-overlapping joins
        """
        
        if type(ground_truth) is np.ndarray:
            result = ground_truth - simulation
            result = (result ** 2).mean()
            result = np.sqrt(result)
            return result

        if type(ground_truth) is list:

            ground_truth = np.nan_to_num(ground_truth)
            simulation   = np.nan_to_num(simulation)

            result = np.asarray(ground_truth) - np.asarray(simulation)
            result = (result ** 2).mean()
            result = np.sqrt(result)

            return result

        df = self.join_dfs(ground_truth, simulation, join=join,
            fill_value=fill_value)


        if len(df.index) > 0:

            if cumulative:
                df['value_sim'] = df['value_sim'].cumsum()
                df['value_gt'] = df['value_gt'].cumsum()
                
            if normed:
                if df['value_gt'].min() > 0:
                    epsilon = 0.001*df[df['value_gt'] != 0.0]['value_gt'].min()
                elif df['value_sim'].min() > 0:
                    epsilon = 0.001*df[df['value_sim'] != 0.0]['value_sim'].min()
                else:
                    epsilon = 1.0
                    
                df['value_sim'] = (df['value_sim'] + epsilon)/(df['value_sim'].max() + epsilon)
                df['value_gt'] = (df['value_gt'] + epsilon)/(df['value_gt'].max() + epsilon)

            if not relative:
                return np.sqrt(((df["value_sim"]-df["value_gt"])**2).mean())
            else:
                iq_range = float(iqr(df['value_gt'].values))

                result = df["value_sim"]-df["value_gt"]
                result = (result ** 2).mean()
                result = np.sqrt(result)

                if iq_range > 0:
                    result = result / iq_range
                else:
                    mean_value = df['value_gt'].mean()
                    if mean_value > 0:
                        result = result / mean_value
                    else:
                        return None

                return result
        else:
            return None


    def r2(self, ground_truth, simulation, join='inner', fill_value=0):
        """
        Metric: r2

        Description: R-squared value between ground truth and simulation

        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column
        simulation - simulation measurement (data frame) with measurement in the
            "value" column
        join - type of join to perform between ground truth and simulation
        fill_value - fill value for non-overlapping joins
        """

        if simulation is None or ground_truth is None:
            return None

        if len(simulation) == 0 or len(ground_truth) == 0:
            return None

        if type(ground_truth) is list:
            ground_truth = np.nan_to_num(ground_truth)
            simulation = np.nan_to_num(simulation)

            ground_truth = ground_truth[np.isfinite(ground_truth)]
            simulation = simulation[np.isfinite(simulation)]

            return np.sqrt(((np.asarray(ground_truth) - np.asarray(simulation)) ** 2).mean())

        ground_truth = ground_truth[np.isfinite(ground_truth.value)]
        simulation = simulation[np.isfinite(simulation.value)]

        df = self.join_dfs(ground_truth,simulation,join=join,fill_value=fill_value).fillna(0)

        if df.empty or len(df.index) <= 1:
            return None
        else:
            return r2_score(df["value_gt"],df["value_sim"])


    def pearson(self, ground_truth, simulation, join='inner', fill_value=0):
        """
        Metric: pearson

        Description: Pearson correlation coefficient between simulation and ground truth

        Inputs:
        ground_truth - ground truth measurement (data frame) with measurement in
            the "value" column
        simulation - simulation measurement (data frame) with measurement in the
            "value" column
        join - type of join to perform between ground truth and simulation
        fill_value - fill value for non-overlapping joins
        """

        df = self.join_dfs(ground_truth,simulation,join=join,fill_value=fill_value)

        if len(df.index) > 0:
            return pearsonr(df["value_gt"],df["value_sim"])
        else:
            return None


    def ks_test(self, ground_truth, simulation):
        """
        Metric: Kolmogorov-Smirnov test.

        Description: Meant for measurements which are continuous or numeric distributions

        Input:
        """

        if simulation is None or len(simulation) == 0 or ground_truth is None or len(ground_truth) == 0:
            return None

        ground_truth, simulation = self.check_data_types(ground_truth, simulation)

        return ks_2samp(ground_truth,simulation).statistic


    def multi_distributions(self, ground_truth, simulation, **kwargs):
        """
        Description:

        Input:
            :ground_truth:
            :simulation:
            :metric:

        Output:
        """
        metric = kwargs.pop('metric')

        cols = [c for c in ground_truth.columns if c != 'value']

        ground_truth = ground_truth.copy()
        simulation = simulation.copy()

        ground_truth['source'] = 'ground_truth'
        simulation['source'] = 'simulation'

        df = pd.concat([ground_truth,simulation])

        grouped = df.groupby(cols)

        metrics = []
        for g, grp in grouped:

            gt = grp[grp['source'] == 'ground_truth']
            sim = grp[grp['source'] == 'simulation']

            if metric == "kl":
                m = self.kl_divergence(gt, sim, **kwargs)
            elif metric == "kl_smoothed":
                m = self.kl_divergence_smoothed(gt, sim, **kwargs)
            elif metric == "js":
                m = self.js_divergence(gt, sim, **kwargs)
            elif metric == 'ks':
                m = self.ks_test(gt,sim, **kwargs)

            if np.isfinite(m):
                metrics.append(m)

        return(np.mean(metrics))


    def spearman(self, ground_truth, simulation, join="inner", fill_value=0):
        """
        Metric: spearman

        Description: measures the strength and direction of association between two ranked variables.
        (nonparametric version of the Pearson product-moment correlation)

        Input:

        """

        df = self.join_dfs(ground_truth, simulation, join=join, fill_value=fill_value)
        if len(df.index) > 1:
            return spearmanr(df["value_gt"], df["value_sim"])[0]
        else:
            return None
