from matplotlib import pyplot as plt

def plot_best():
    y = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x_0025 = [100, 99.966, 99.621, 99.204, 97.447, 94.902, 91.480, 90.267, 89.911, 81.497, 0]
    x_005 = [100, 99.966, 99.134, 97.834, 96.215, 93.611, 90.452, 88.370, 82.412, 49.605, 0]
    x_01 = [100, 99.897, 99.551, 97.863, 94.970, 90.141, 83.496, 72.517, 50.126, 15.818, 0]
    x_02 = [100, 97.751, 96.151, 91.542, 82.568, 71.595, 55.835, 36.548, 17.036, 4.578, 0]
    plt.plot(y, x_0025, label="2.5ms")
    plt.plot(y, x_005, label="5ms")
    plt.plot(y, x_01, label="10ms")
    plt.plot(y, x_02, label="20ms")
    plt.ylabel('IOU F1 [%]')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid()
    plt.show()

def plot_sliding_window():
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    logistic_regression = [45.304, 42.089, 30.911, 20.842, 12.943, 7.396, 2.568, 0.616, 0]
    svm = [50.126, 48.864, 39.289, 28.106, 16.802, 8.045, 3.666, 0.916, 0]
    random_forest = [61.517, 60.056, 50.094, 37.695, 26.112, 14.664, 6.528, 1.987, 0]
    gradient_boosting = [66.464, 64.957, 55.849, 44.336, 30.591, 18.052, 9.249, 3.112, 0]
    plt.plot(y, logistic_regression, label="Logistic regression", marker='.')
    plt.plot(y, svm, label="SVM", marker='.')
    plt.plot(y, random_forest, label="Random forest", marker='.')
    plt.plot(y, gradient_boosting, label="Gradient boosting", marker='.')
    plt.ylabel('IOU F1 [%]')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid()
    plt.locator_params(nbins=10)
    plt.show()

def plot_test_set():
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    crnn = [88.005, 87.108, 84.984, 80.790, 75.491, 67.705, 57.075, 39.878, 12.255, 0]
    rnn = [88.319, 86.566, 83.632, 80.900, 76.104, 71.420, 63.989, 51.004, 23.829, 0]
    plt.plot(y, crnn, label="CRNNFinal", marker='.')
    plt.plot(y, rnn, label="RNNFinal", marker='.')
    plt.ylabel('IOU F1 [%]')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid()
    plt.locator_params(nbins=10)
    plt.show()

def plot_performance():
    y = [400, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    crnn = [3.533,5.619,8.623,11.645,14.849,17.885, 21.139, 23.829, 26.893, 29.647, 33.918]
    rnn = [4.170,6.665,10.558,14.334,18.075,21.975,25.743,29.301,32.922,36.691,40.555]
    plt.plot(y, crnn, label="CRNNFinal", marker='.')
    plt.plot(y, rnn, label="RNNFinal", marker='.')
    plt.ylabel('Inference time [s]')
    plt.xlabel('Recording length [s]')
    plt.legend()
    plt.grid()
    plt.locator_params(nbins=10)
    plt.show()

if __name__ == '__main__':
    plot_performance()