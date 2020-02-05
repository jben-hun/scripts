#! /usr/bin/python3

import argparse
from sys import exit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    print(args.file)

if __name__ == '__main__':
    main()
