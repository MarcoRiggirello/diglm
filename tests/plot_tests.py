""" Unit tests for the plot_utils module """
import unittest
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plot_utils import multi_sns_plot, make_gif


class TestPlotUtils(unittest.TestCase):
    """
    Tests on plot_utils
    """

    def test_gif(self):
        """
        Tests make_gif function. Checks that a file is created.
        Produces a gif of a projectile motion.
        """

        time = np.linspace(0., 2., 100)
        x_var = time
        y_var = - 5 * (time ** 2)
        name_fig = []
        for i in range(100):
            plt.title('Projectile motion')
            plt.xlim(0., 2.5)
            plt.ylim(-20., 1.)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.errorbar(x_var[:i], y_var[:i], color='blue')
            plt.errorbar(x_var[i], y_var[i], fmt='o', color='orange')
            plt.savefig(f'projectile{i}.png')
            name_fig.append(f'projectile{i}.png')
            plt.clf()
        make_gif(name_fig, output_gif='my_gif.gif', duration=0.03)
        # Tests that the file has been created
        self.assertTrue(os.path.isfile('my_gif.gif'))
        # Tests that images have been deleted
        self.assertFalse(os.path.isfile(name_fig[0]))

    def test_plot(self):
        """
        Tests multi_sns_plot. Checks that a plot is actually created
        """

        pingu = sns.load_dataset('penguins')
        pingu1 = pingu.drop(pingu[pingu.species != 'Gentoo'].index)
        pingu2 = pingu.drop(pingu[pingu.species == 'Gentoo'].index)
        df_list = [pingu1, pingu2]
        namefig = 'penguins_plot.png'
        multi_sns_plot(df_list,
                       x_var='body_mass_g',
                       y_var='flipper_length_mm',
                       names=['Gentoo specie', 'Other species'],
                       title=namefig,
                       kind='kde')
        self.assertTrue(os.path.isfile(namefig))


if __name__ == '__main__':
    unittest.main()
