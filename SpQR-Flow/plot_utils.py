""" Module with plot utilities """
import logging

import imageio


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

    if isinstance(input_im, str):
        fig_list = imageio.mimread(input_im)
    else:
        fig_list = [imageio.imread(fig) for fig in input_im]
    if not output_gif.find('.gif'):
        output_gif = f'{output_gif}.gif'
        logging.info(f'output filename is {output_gif}.')
    imageio.mimsave(output_gif, fig_list, duration=duration)
