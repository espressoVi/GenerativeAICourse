#!/usr/bin/env python
from model import Falcon

def main():
    agent = Falcon()
    print(f"\n{'_'*50}\n\tFalcon 7B Instruct\n{'_'*50}\n")
    while True:
        sequence = input(">>> ")
        result = agent(sequence)
        print(result)
        print(f"\n{'_'*50}\n")
    
if __name__ == "__main__":
    main()
