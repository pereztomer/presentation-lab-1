import matplotlib.pyplot as plt

# Your two lists
list1 = [4, 6, 9, 3, 7, 1, 8, 2, 5]
list2 = [1, 3, 5, 7, 9, 11, 13, 15, 17]



from data_utils import plot_graph

plot_graph(list1, list2, 'train', 'test', 'train/test vs epoch')