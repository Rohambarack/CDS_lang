import os
os.system("pip install numpy")
import numpy as np

def main():
    filename = os.path.join("..","in","sample-data-01.csv")
    data = np.loadtxt(filename, delimiter= ",")

    print(data)

#
if __name__ == "__main__":
    main()