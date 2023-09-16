# Question answering on the CoQA dataset.

Implementation of training and evaluation of BERT-base for the CoQA dataset.
____________________________________________________

## Task description

For this session, your mission is to write a real-time interactive inference routine. A context paragraph will be provided to you, based on which you have to
  - Take user questions based on the paragraph as inputs.
  - Combine the user questions with the story context and create the model input.
  - Run inference on the model and generate output logits.
  - Convert the output logits into textual output.
  - Provide the output to the user.
  - Allow for further turns of questioning (future question turns might depend on answers of previous turns - i.e. conversational)
  - You may use any resource available to you.

Stories can be found in **"./stories"** directory. Solutions will be qualitatively evaluated.

____________________________________________________

## Instructions to run.

  - The **CoQA dataset** is to be downloaded and place in **"./data"**, [train files](https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json) and [validation files](https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json). You should have the following files.

  ```
  ./data/coqa-dev-v1.0.json
  ./data/coqa-train-v1.0.json
  ```
  - The config file **"config.toml"** contains necessary settings and global variables. Modify if needed.

  - **Training and predictions** is handled using the "main.py" file.

  ```
  python main.py 
  ```

  - **Evaluation** [Optional]
    Following eval you will have a predictions.json file at the provided directory. Then run

    ```
    python processors/eval.py
    ```
____________________________________________________

## Files.
  - main.py                     -> Runs training and evaluation.
  - model.py                    -> BERT-base-uncased + QA model implementation.
  - train.py                    -> Implements training and evaluation procedures.
  - config.toml                 -> Configuration file - global variables, settings for the program.
  - README.md                   -> This file.
  - processors/
    - coqa.py                   -> Implements data cleaning, preprocessing, tokenization, etc
    - eval.py                   -> Official evaluation script for CoQA dataset.
    - metric.py                 -> Converts output logits to text i.e. **Post-processing.**
  - data/
    - coqa-dev-v1.0.json        -> Validation set CoQA dataset.
    - coqa-train-v1.0.json      -> Train set CoQA dataset.
  - stories/                    -> **Contains stories used for live inference.**
____________________________________________________
