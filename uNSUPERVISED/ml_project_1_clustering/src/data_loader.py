import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","Wholesale customers data.csv")

def load_data(path):
    return pd.read_csv(DATA_PATH)

def visualize(x,y,visualTitle=None, visualXlabel = None, visualYlabel=None, label=None):
    plt.plot(x,y,c=label)
    if visualTitle is not None:
        plt.title(visualTitle)
    if (visualXlabel is not None) and (visualYlabel is not None):
        plt.xlabel(str(visualXlabel))
        plt.ylabel(str(visualYlabel))
    plt.show()

# x = [x for x in range(1,10)]
# y = [y**2 for y in range(1,10)]
# # print(x,y)
# visualize(x,y,"X vs Y","X","Y")