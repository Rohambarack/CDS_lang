import os
os.system("pip install numpy")
import argparse 
import numpy as np

def file_loader():
    parser = argparse.ArgumentParser(description = "Loading and printing an array")

    parser.add_argument("--input",
                        "-i",
                        required = True,
                        help = "Filepath to csv for loading and printing")
    args = parser.parse_args()

    return args

def main():
    args = file_loader()
    filename = os.path.join("..","in",args.input)
    data = np.loadtxt(filename, delimiter= ",")

    print(data)

#
if __name__ == "__main__":
    main()