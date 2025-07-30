import numpy as np
import matplotlib.pyplot as plt

# Dummy data for plotting
x_truth = np.array([0, 1, 2, 3])
y_truth = np.array([0, 1, 0, 1])
x_pred = np.array([0, 1, 2, 3])
y_pred = np.array([0.1, 0.9, 0.2, 1.1])

def plot(x_truth, y_truth, x_pred, y_pred):
    plt.plot(x_truth, y_truth, linestyle = '--', marker = 'o', color = 'blue', markersize = 5, markerfacecolor = 'black', label = 'Ground Truth')
    plt.plot(x_pred, y_pred, linestyle = '--', marker = 'o', color = 'red', markersize = 5, markerfacecolor = 'black', label = 'Prediction')

    for i, (xt, yt) in enumerate(zip(x_truth, y_truth)):
        plt.annotate(str(i), (xt, yt), textcoords = "offset points", xytext = (5,5), ha = 'center', color = 'blue')
    for i, (xp, yp) in enumerate(zip(x_pred, y_pred)):
        plt.annotate(str(i), (xp, yp), textcoords = "offset points", xytext = (5,-10), ha = 'center', color = 'red')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Robot Localization")
    plt.legend()

    plt.show()

def graph(samples, losses):
    plt.figure()
    plt.plot(samples, losses, marker='o', linestyle='-')
    plt.xlabel("Samples")
    plt.ylabel("Loss")
    plt.title("Samples vs Loss Curve")
    plt.grid(True)
    plt.show()
    

def main():
    # plot(x_truth, y_truth, x_pred, y_pred)
    graph([15, 25, 35, 100, 200, 300, 400, 500, 600], [0.0097, 0.0461, 0.1402, 0.0645, 0.1534, 0.1233, 0.0507, 0.1729, 0.0755])

if __name__ == "__main__":
    main()
