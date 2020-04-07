1. REQUIREMENTS
=========================================================================================
The system must meet the following requirements in order to run the program:
- Python 3.7
- Pandas 0.90 or newer
- Scikit-learn 0.21 or newer


2. PREPROCESS.PY script
=========================================================================================
This script is provided with the purpose of meeting project requirements to generate
a pre-processed training dataset in CSV and ARFF formats for Weka.

Weka is NOT used in this project.

Usage: python3 preproces.py


3. TRAINER.PY application
=========================================================================================
The program was tested on 'red.eecs.yorku.ca' and executed properly without the need to
install any additional libraries.

IMPORTANT: Please ensure that the 'test3.csv' and 'train3.csv' datasets are located in the same directory
as the program (trainer.py) or change the train_dataset_path and test_dataset_path
configuration variables accordingly.

Usage: python3 trainer.py

APPENDIX A: Sample run on red.eecs.yorku.ca
==========================================================================================
red 304 % ls
EECS 4412 Project.pdf  preprocess.py  readme.txt  stop_words.txt  test3.csv  train3.csv  trainer.py
red 305 % python3 trainer.py
==== TRAINING DATASET CLASSES ===
=== Training a Linear SVM Classifier ===
=== Training a Logistic Regression Classifier ===
=== Training a Random Forest Classifier ===

[ ... redacted ...]

All tasks completed.
red 306 %
