from lang_s_08 import Person
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