import matplotlib.pyplot as plt
from random import randint
from matplotlib.animation import FuncAnimation
from easy_trilateration import model
from easy_trilateration.graph import *  
import matplotlib

def plot_step_with_distances(historyStep: model.Trilateration):
    x_values = []
    y_values = []
    x_values.append(historyStep.result.center.x)
    y_values.append(historyStep.result.center.y)

    for item in historyStep.sniffers:
        create_point(item.center)
        create_circle(item, target=False)

    create_circle(historyStep.result, target=True)
    plt.plot(x_values, y_values, 'blue', linestyle='--')
    plt.show()

