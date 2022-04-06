""" Unit tests for the plot_utils module """
import unittest
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plot_utils import multi_sns_plot, make_gif


class TestPlotUtils(unittest.TestCase):
    def test_gif(self):
        """
        Tests make_gif function. Checks that a file is created.
        Produces a gif of a projectile motion.
        """

        t_ = np.linspace(0., 1., 100)
        x_ = t_
        y_ = - 5 * (t_ ** 2)
        name_fig = []
        for i in range(100):
            plt.figure(1)
            plt.xlim(0., 1.5)
            plt.ylim(0., -5)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.errorbar(x_[i], y_[i], fmt='o')
            plt.savefig(f'projectile{i}.png')
            name_fig.append(f'projectile{i}.png')
        make_gif(name_fig, output_gif='my_gif.gif', duration=0.03)
        self.assertTrue(os.path.isfile('my_gif.gif'))
        self.assertFalse(os.path.isfile('my_gif'))
        #os.remove('my_gif.gif')

        
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
                       figname=namefig,
                       kind='kde')
        self.assertTrue(os.path.isfile(namefig))


if __name__ == '__main__':
    unittest.main()
