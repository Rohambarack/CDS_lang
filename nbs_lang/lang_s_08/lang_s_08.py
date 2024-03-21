# 4 add argparse arguments
import argparse

def person_parser():
    parser =  argparse.ArgumentParser(description='Define human class')
    parser.add_argument('--name',
                        "-n",
                        required = True,
                        help='name of person')
    parser.add_argument('--age', help='age of person')
    parser.add_argument("--likes",
                        "-l")

    args = parser.parse_args()
    return args

# 6 make classes
class Person:

    species = "Homo Sapiens"

    def __init__(self, name, age, likes):
        self.name = name
        self.age = age
        self.likes = likes
    # a method is a function bound to a class
    # 1 use functions
    def hello(self):
        print("hello, " + self.name)

    def preferences(self):
        print("I like " + self.likes)

# 2 use functions in a main function
def main():
    #good idea to use uppercase name for classes
    
    arguments = person_parser()

    person_1 = Person(arguments.name,arguments.age,arguments.likes)
    print(person_1.age)
    person_1.hello()
    person_1.preferences()

    like = person_1.likes
    print(like)

# 3 define main to run only when script is run directly
if __name__ == "__main__":
    main()


# 5 make a virtual environment, save used packages, make a setup txt for loading
