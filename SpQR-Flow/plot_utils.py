""" Module with plot utilities """
from functools import wraps
import logging

import imageio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_gif(input_im, output_gif='my_gif.gif', duration=0.5):
    """
    Function to create a gif out of a list of figures.

    :param input_im: file names of images or directory name containing the images to compose the animated gif.
    :type input_im: list[str] or array-like or str
    :param output_gif: Optional (Default=\'my_gif.gif\'). Name of the output gif file. If the name doesn't end with the .gif extension, it is added automatically.
    :type output_gif: str
    :param duration: Optional (Default=0.5) define the time interval (in seconds) between frames.
    :type duration: float

    """

    #if isinstance(input_im, str):
    #    fig_list = imageio.mimread(input_im)
    #else:
    fig_list = [imageio.imread(fig) for fig in input_im]
    imageio.mimsave(output_gif, fig_list, duration=duration)
                        

def multi_sns_plot(df_list,
                   x_var=None,
                   y_var=None,
                   names=None,
                   figname=None,
                   display=False,
                   **kwargs):
    """
    Plots more than one dataframe on the same canvas.
    :param df_list: list of dataframes to be plotted.
    :type df_list: list[`pandas.DataFrame`]
    :param x_var: dataframe column label plotted on x axis.
    :type x_var: str
    :param y_var: dataframe column label plotted on y axis.
    :type y_var: str
    :param names: labels for the dataframes plotted.
    :type names: list[str]
    :param **kwargs: Extra key-word arguments to be passed to sns. 
    """

    # Create a new dataframe from concatenation of the datasets in df_list.
    # The datasets are assigned a label according to `names` entries.
    df_list = [df.assign(dataset=names[i]) for i, df in enumerate(df_list)]
    concatenated = pd.concat(df_list,
                             ignore_index=True, # avoids duplicated indices 
                             )
    # Plotting
    sns.displot(data=concatenated,
                x=x_var, y=y_var,
                hue='dataset',
                **kwargs)
    if figname is not None:
        plt.savefig(figname)
    if display:
        plt.show()
