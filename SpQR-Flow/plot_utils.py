""" Module with plot utilities """
import os

import numpy as np
import imageio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def make_gif(input_im,
             output_gif='my_gif.gif',
             duration=0.5,
             keep_im=False):
    """
    Function to create a gif out of a list of figures.

    :param input_im: file names of images to compose the animated gif.
    :type input_im: list[str]
    :param output_gif: Optional (Default=\'my_gif.gif\'). Name of the output gif file.
    :type output_gif: str
    :param duration: Optional (Default=0.5). Define the time interval (in seconds) between frames.
    :type duration: float
    :param keep_im: Optional (Default=False). If True, the images used to create the gif are kept in memory; otherwise they are deleted.
    :type keep_im: bool
    :raise OSError: Incorrect file(s) name.
    """

    # Check for correct paths
    path_bool = np.array([os.path.exists(os.path.abspath(im))
                          for im in input_im])
    if not path_bool.any():
        raise OSError

    fig_list = [imageio.imread(fig) for fig in input_im]
    imageio.mimsave(output_gif, fig_list, duration=duration)
    # Eliminating images to save space
    if not keep_im:
        for fig in input_im:
            os.remove(fig)


def multi_sns_plot(df_list,
                   x_var=None,
                   y_var=None,
                   names=None,
                   title=None,
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
    :param title: Optional (Default=None). Title of the final plot.
    :type title: str
    :param display: Optional (Default=False). Open the resulting plot.
    :type display: bool
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
    if title is not None:
        plt.savefig(title)
    if display:
        plt.show()
