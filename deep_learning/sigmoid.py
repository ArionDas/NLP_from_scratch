import numpy as np

def sigmoid(x) : 
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    x = np.array([-10, 2, 3, 4, 500])
    print(sigmoid(x))