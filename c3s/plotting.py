import matplotlib.pyplot as plt
import mplcyberpunk
import seaborn as sns
from typing import List


class PlottingMixin:

    def plot_probability_evolution(self, molecules: List[str], data=None, save=None):
        """"""

        molecules = sorted(molecules)
        if data is None:
            data = self.calculate_marginal_probability_evolution(molecules)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.despine()
        #plt.style.use('cyberpunk')
        #mplcyberpunk.make_lines_glow(ax)
        title = ''
        for molec in molecules:
            title += f'n_{{{molec}}},'
        plt.ylabel('probability', fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.title(rf'$p({title[:-1]})$ evolution', fontsize=18)
        for point in data:
            plt.plot(data[point], label=f'{point}')
        plt.legend(fontsize=12)

        plt.show()

    def plot_mutual_information_evolution(self, X, Y, data=None):

        if data is None:
            data = self.calculate_mutual_information(X, Y)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.despine()
        #plt.style.use('cyberpunk')
        #mplcyberpunk.make_lines_glow(ax)
        title1 = ''
        title2 = ''
        for x in X:
            title1 += f'n_{{{x}}},'
        for y in Y:
            title2 +=  f'n_{{{y}}},'
        plt.ylabel('mutual information (nats)', fontsize=14)
        plt.xlabel('time', fontsize=14)
        plt.title(rf'MI between $p({title1[:-1]})$ and $p({title2[:-1]})$', fontsize=18)
        plt.plot(data)

        plt.show()

def plot_benchmark_results(filename):
    pass

