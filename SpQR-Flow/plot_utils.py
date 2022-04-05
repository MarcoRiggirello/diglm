""" Module with plot utilities """
import seaborn as sns
import imageio
import logging

def make_gif(input_im, output_gif='my_gif.gif', duration=0.5):
    """
    Function to create a gif out of a list of figures.

    :param input_im: file names of images or directory name
    with the images to compose the animated gif.
    :type input_im: list[str] or any iterable containing
    text strings or a str with the path to a directory where 
    files are stored.
    :param output_gif: name of the output gif file. If the
    name doesn't end with the .gif extension, it is added
    automatically.
    :type output_gif: str
    :param duration: Optional define the time interval between
    frames.
    :type duration: float

    """
    if type(input_im) == str:
        fig_list = [fig for fig in imageio.mimread(input_im)]
    else:
        fig_list = [imageio.imread(fig) for fig in input_im]
    if not output_gif.find('.gif'):
        output_gif = f'{output_gif}.gif'
        logging.info(f'output filename is {output_gif}'.)
    imageio.mimsave(output_gif, fig_list, duration=duration)
