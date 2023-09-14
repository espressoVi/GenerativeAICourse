#!/usr/bin/env python
from model import Falcon

def main():
    sequence = "The quick brown fox jumps over the lazy dog. Who does the quick yellow fox jump over?"
    model = Falcon()
    print(model.infer(sequence))
    
if __name__ == "__main__":
    main()
