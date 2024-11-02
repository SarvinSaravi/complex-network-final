import numpy as np
import pandas as pd

# a = np.array([[-5.349, -4.349], [-2.349, -1.349], [-3.349, -0.349], [2.651, 0.651]], dtype='float64')
# print(a)
#
# b = a[a[:, 0].argsort()]
# print(b)

len_val = 2304

# df = pd.DataFrame({
#     ('pixel ' + str(i)) for i in range(5)
# })

# s = [12, 34, 44, 56, 7]
# i = [str(i) for i in range(5)]
# print(i)
#
# data = {str(i): [] for i in range(5)}
#
# df = pd.DataFrame(data)

data2 = [[[2, 2], [2, 5], [3], [23, 66], [455, 990, 2]],
         [[2, 3], [2, 5], [3, 6], [23], [455, 990, 2, 7]]]

# x = [str(len(u))] for u in data2

# data2 = np.array(data2, dtype=object)

# for item in data2:
#     item: object = len(j) for j in item

df2 = pd.DataFrame(data2)

# print(df.head())
print(df2.head())

# df2.to_csv('test.csv')


#################################################################################################33333


#
# # import dataset
# df = pd.read_csv("politician_edges.csv")
# df.head()
#
# # create Graph
# G = nx.from_pandas_edgelist(df, "node_1", "node_2", create_using=nx.Graph())
# print(len(G))
#
# # train model and generate embedding
# ################## for main dataset
# # model = DeepWalk(walk_number=1000, walk_length=10, dimensions=20)
# ################## for this dataset
# model = DeepWalk(walk_length=100, dimensions=64, window_size=5)
# model.fit(G)
# embedding = model.get_embedding()
#
# # print Embedding shape
# print(embedding.shape)
# # take first 100 nodes
# nodes = list(range(100))
#
# # for convert float numbers to integer
# f = lambda i: int(i - i % (1 | ~(i > 0)))
#
# df_data = []
#
# # plot nodes graph
# def plot_nodes(node_no):
#     X = embedding[node_no]
#
#     # use PCA for dimension reduction
#     pca = PCA(n_components=2)
#     pca_out = pca.fit_transform(X)
#     print(pca_out.shape)  # (100,2)
#
#     # plot the points
#     # plt.figure(figsize=(15, 10))
#     scat = plt.scatter(pca_out[:, 0], pca_out[:, 1], marker='o')
#     for i, node in enumerate(node_no):
#         plt.annotate(node, (pca_out[i, 0], pca_out[i, 1]))
#     plt.xlabel('Label_1')
#     plt.ylabel('Label_2')
#
#     # get the points coordinate
#     plot_data = scat.get_offsets()
#     xdata = plot_data[:, 0]
#     ydata = plot_data[:, 1]
#
#     x_max = f(xdata.max())
#     x_min = f(xdata.min())
#
#     y_max = f(ydata.max())
#     y_min = f(ydata.min())
#
#     x_step = (x_max - x_min) / 48
#     y_step = (y_max - y_min) / 48
#
#     plot_data = plot_data[plot_data[:, 0].argsort()]
#
#     plt.show()
#
#     all_point = []
#
#     for x in np.arange(x_min, x_max, x_step, dtype='float64'):
#         for y in np.arange(y_min, y_max, y_step, dtype='float64'):
#             # print('we are in x : ', x, 'and y: ', y)
#             count = []
#             for i, j in plot_data:
#                 if x <= i < (x + x_step):
#                     if y <= j < (y + y_step):
#                         count.append([i, j])
#             print(count)
#             all_point.append(len(count))
#
#     # print(len(all_point))
#     # print(all_point)
#
#     return all_point
#
#
#     # bins = np.arange(0, 48, 8)
#     # h = plt.hist2d(pca_out[:, 0], pca_out[:, 0], bins=(bins, bins))
#     # plt.colorbar(h[3])
#     # plt.show()
#
#
# all_p = plot_nodes(nodes)
# df_data.append(all_p)
# df_data.append(all_p)
#
# df2 = pd.DataFrame(df_data)
#
# print(df2.head())


