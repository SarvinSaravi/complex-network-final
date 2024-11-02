import networkx as nx
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from generate_graph import GraphGenerator


class Representation:
    def __init__(self):
        self.graph_generator = GraphGenerator(graph_count=10)
        self.graph_list = list(self.graph_generator.generate_graphs())

        self.df_data = []

        self.f = lambda i: int(i - i % (1 | ~(i > 0)))

    def main_function(self):
        self.selecting()

        return self.end_process()

    def selecting(self):
        for i in range(len(self.graph_list)):
            # walk through graph list
            nds, embed = self._graph_to_latent_rep(self.graph_list[i])
            # check if latent representation of graph not be blank - if it's blank graph is not appropriate to computing
            if len(nds) > 0 and len(embed) > 0:
                p_data = self._pca_and_plot(embed, nds)
                pixels = self._plot_to_picture(p_data)
                self.df_data.append(pixels)

    @staticmethod
    def _graph_to_latent_rep(graph_edge_dataframe):
        # check if possible to create DataFrame - if not, return blank
        if type(graph_edge_dataframe) == pd.core.frame.DataFrame:
            df = pd.DataFrame(graph_edge_dataframe, index=None, columns=None)

            G = nx.from_pandas_edgelist(df, "source", "target", create_using=nx.Graph())

            # train model and generate embedding
            model = DeepWalk(walk_number=1000, walk_length=10, dimensions=20)
            model.fit(G)
            embedding = model.get_embedding()

            nodes = list(range(len(G)))

            return nodes, embedding
        else:
            return [], []

    @staticmethod
    def _pca_and_plot(embedding, node_no):
        X = embedding[node_no]

        # use PCA for dimension reduction
        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(X)

        # plot the points
        scat = plt.scatter(pca_out[:, 0], pca_out[:, 1], marker='o')
        for i, node in enumerate(node_no):
            plt.annotate(node, (pca_out[i, 0], pca_out[i, 1]))
        plt.xlabel('Label_1')
        plt.ylabel('Label_2')

        plt.show()

        # get the points coordinate
        plot_data = scat.get_offsets()

        return plot_data

    def _plot_to_picture(self, plot_data):
        xdata = plot_data[:, 0]
        ydata = plot_data[:, 1]

        x_max = self.f(xdata.max())
        x_min = self.f(xdata.min())

        y_max = self.f(ydata.max())
        y_min = self.f(ydata.min())

        x_step = (x_max - x_min) / 48
        y_step = (y_max - y_min) / 48

        plot_data = plot_data[plot_data[:, 0].argsort()]

        all_point = []

        for x in np.arange(x_min, x_max, x_step, dtype='float64'):
            for y in np.arange(y_min, y_max, y_step, dtype='float64'):
                count = []
                for i, j in plot_data:
                    if x <= i < (x + x_step):
                        if y <= j < (y + y_step):
                            count.append([i, j])
                all_point.append(len(count))

        return all_point

    def end_process(self):

        df2 = pd.DataFrame(self.df_data)

        df2.to_csv('file.csv')

        return df2


rep = Representation()
print(rep.main_function())
