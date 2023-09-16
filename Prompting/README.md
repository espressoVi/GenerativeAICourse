# Prompting tutorial.

This module contains code to generate text output from Falcon-7b (instruct finetuned) model.
____________________________________________________

## Task description.

You have been provided with code that generates output from an LLM model. The current implementation works like a chat bot (with only one turn history) as an example. There are several provided tasks in the directory **"./tasks"**. You are required to 
  - Solve as many tasks you can with LLM prompting.
  - Try to implement several prompting strategies (bonus points for CoT-SC or TOT implementation).
  - Further documentation can be found in the **references**.
____________________________________________________

## Instructions for running.

  - Just run the main file.
  ```
  python main.py
  ```
____________________________________________________

## Files
  - main.py                     -> I/O routines.
  - model.py                    -> Inference pipeline.
  - config.toml                 -> Configuration file - global variables, settings for the program.
  - README.md                   -> This file.
  - tasks/                      -> Folder containing tasks.
____________________________________________________
