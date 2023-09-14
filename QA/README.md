# Question answering on the CoQA dataset.

Implementation of training and evaluation of BERT-base for the CoQA dataset.
____________________________________________________

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

## Task description

For this session, your mission is to write a real-time interactive inference routine. A story will be provided to you. You have to
  - Take user questions as inputs.
  - Combine the user questions with the story context and create the model input.
  - Run inference on the model and generate output logits.
  - Convert the output logits into textual output.
  - Provide the output to the user.
  - Allow for further turns of questioning (future question turns might depend on answers of previous turns - i.e. conversational)
  - You may use any resource available to you.

____________________________________________________
