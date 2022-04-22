""" Module with plot utilities """
import os

import numpy as np
import imageio


def make_gif(input_im,
             output_gif='my_gif.gif',
             duration=0.5,
             keep_im=False):
    """
    Function to create a gif out of a list of figures.

    :param input_im: file names of images to compose the animated gif.
    :type input_im: list[str]
    :param output_gif: Optional (Default="my_gif.gif"). Name of the output gif file.
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
