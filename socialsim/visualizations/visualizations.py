
import os

from . import charts
from . import transformer

from .visualization_config import measurement_plot_params


def generate_plot(ground_truth=None,
                  simulation=None,
                  measurement_name='',
                  logx = False,
                  logy = False,
                  plot_dir='plots/',
                  show=False,
                  max_plots=10,
                  ymin=None,
                  ymax=None):
    """
    Generates a visualization of a given measurement and either displays to
    screen or saves the plot
    Inputs:
        simulation - Output of measurement function on the simulation data
        ground_truth - Output of measurement function on the ground truth data
        measurement_name - The name of the measurement being plotted
        show - Boolean indicating whether to display plots to screen
        plot_dir - Directory to save plots. If empty string, the plots will not
            be saved.
    """
    if measurement_name in measurement_plot_params and not (simulation is None and ground_truth is None) and \
            not ((not simulation is None and len(simulation)==0) and (not ground_truth is None and len(ground_truth)==0)):

         # get plotting parameters for given measurement
         params = measurement_plot_params[measurement_name]

         #print(params)

         # keys are the IDs for nodes and communities to extract individual 
         # measurements from the dictionary
         keys = ['']
         if 'plot_keys' in params:
             try:
                 keys = simulation.keys()
             except:
                 keys = ground_truth.keys()

         # only generate limited number of plots for specific nodes/communities 
         # if showing to screen
         keys = list(keys)
         #print(keys)
         if show and len(keys) > max_plots:
             keys = keys[:max_plots]


         #loop over individual nodes or communities
         for key in keys:
             #preprocess measurement output to prepare for plotting
             if key != '':
                 df = transformer.to_DataFrame(params['data_type'])(sim_data=simulation, ground_truth_data=ground_truth,key=key)
             else:
                 df = transformer.to_DataFrame(params['data_type'])(sim_data=simulation, ground_truth_data=ground_truth)

             #print(key,df)
             if not df is None:
                 for p in params['plot']:
                     #generate plot
                     fig = charts.chart_factory(p)(df,params['x_axis'],params['y_axis'], (key + ' ' + measurement_name).lstrip(),**params)

                     if logx:
                         for ax in fig.get_axes():
                             ax.set_xscale("log")
                     if logy:
                         for ax in fig.get_axes():
                             ax.set_yscale("log")

                     if not ymin is None:
                         for ax in fig.get_axes():
                             ax.set_ylim(ymin=ymin)
                     if not ymax is None:
                         for ax in fig.get_axes():
                             ax.set_ylim(ymax=ymax)

                     if show and not fig is None:
                         charts.show_charts()

                     if plot_dir != '' and not fig is None:
                         if not os.path.exists(plot_dir):
                             os.mkdir(plot_dir)
                         if key != '':
                             charts.save_charts(fig,plot_dir + '/' + measurement_name + '_' + key.replace('/','@') + '.png')
                         else:
                             charts.save_charts(fig,plot_dir + '/' + measurement_name + '.png')

