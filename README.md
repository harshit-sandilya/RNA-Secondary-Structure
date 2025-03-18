# RNA Secondary Structure Prediction

This repository provides various computational approaches for predicting RNA secondary structures, utilizing machine learning (ML), deep learning (DL), and reinforcement learning (RL) techniques.

## Overview

RNA secondary structure prediction is crucial for understanding RNA functionality and interactions. Traditional methods often rely on thermodynamic models, but recent advancements have incorporated machine learning to enhance prediction accuracy. This repository explores multiple methodologies:

- **ML.py**: Implements machine learning algorithms for RNA secondary structure prediction.
- **DL.py**: Utilizes deep learning architectures to model RNA folding patterns.
- **RL.py**: Applies reinforcement learning strategies to predict RNA structures.

## Repository Structure

- **data/**: Contains datasets used for training and testing the models.
- **logs/**: Stores log files generated during model training and evaluation.
- **results/**: Includes the models applied and performance metrics.
- **utils/**: Utility scripts and helper functions to support the main algorithms.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: This file, providing an overview of the repository.
- **requirements.txt**: Lists the Python dependencies required to run the scripts.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/harshit-sandilya/RNA-Secondary-Structure.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd RNA-Secondary-Structure
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Make the required directories
   ```bash
   mkdir data/csv
   mkdir data/final
   ```
5. Preprocess the raw data to get csv formats
   ```bash
   python data/preprocess.py
   python data/combine.py
   ```

## Usage

Each script corresponds to a different modeling approach:

- **ML.py**: Run this script to apply machine learning models.
- **DL.py**: Execute this script for deep learning-based predictions.
- **RL.py**: Use this script to leverage reinforcement learning techniques.

Ensure that the `data/` directory contains the necessary datasets before running any script. The logs will be saved in the `logs/` directory.

## References

The repository contains three major datasets for it's training and testing and allows users to combine these datsets to get the desired output. You can read about the datasets from the links below:

- [bpRNA: large-scale automated annotation and analysis of RNA secondary structure](https://pmc.ncbi.nlm.nih.gov/articles/PMC6009582/)
- [RNA STRAND: The RNA Secondary Structure and Statistical Analysis Database](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-340)
- [Review of machine learning methods for RNA secondary structure prediction](https://www.nature.com/articles/s41467-019-13395-9)
