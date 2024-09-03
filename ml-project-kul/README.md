# Multi-Agent Learning in Canonical Games and Dots-and-Boxes

## Overview

Welcome to the repository for our project on Multi-Agent Learning in Canonical Games and Dots-and-Boxes. This project aims to develop intelligent agents capable of playing the Dots-and-Boxes game at a competitive level using advanced machine learning and game theory techniques. The report provides detailed insights into our methodologies, experiments, and results.

## Repository Structure

The repository is structured as follows:

- **`.github/`**: GitHub-related files and workflows.
- **`.idea/`**: Project configuration files for IDE.
- **`task2/`**: Contains code and plots related to Task 2.
  - **`agents_and_plots.py`**: Code for agents and plotting results for Task 2.
- **`task3/`**: Contains code related to Task 3, focusing on the Minimax algorithm.
  - **`chains_util.py`**: Utilities for handling chains in the game.
  - **`main.py`**: Main script for Task 3.
  - **`minimax_chains.py`**: Minimax algorithm implementation with chain strategy.
  - **`minimax_symmetry.py`**: Minimax algorithm implementation with symmetry optimizations.
  - **`minimax_template.py`**: Template for Minimax algorithm.
  - **`minimax_transposition.py`**: Minimax algorithm implementation with transposition tables.
- **`task4/`**: Contains all the code related to Task 4 of the project, including training, tuning, and benchmarking the agents.
  - **`agent/`**: Agent implementations.
  - **`benchmarking/`**: Scripts for benchmarking models with variable board sizes.
  - **`benchmarks/`**: Benchmark results.
  - **`play_tournament/`**: Code to play tournaments between different agents.
  - **`submit_files/`**: Final versions of the agents used in the project.
  - **`training/`**: Code for training the AlphaZero and GNN models.
  - **`tuning/`**: Code for hyperparameter tuning of different models.
- **`.gitignore`**: Git ignore file.
- **`main.py`**: Main script to interact with the project.
- **`README.md`**: Project documentation (this file).
- **`requirements.txt`**: List of dependencies required to run the code.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

You can interact with the main functionality of the project through the `main.py` script. This script provides an interface to run benchmarks, tune models, train models, and play tournaments.

### Running the Script

To run the main script, use the following command:

```sh
./main.py
```

### Script Options

Upon running the script, you will be prompted to choose from the following actions:

1. **Run the benchmarks**: Benchmark different agents against a random agent.
2. **Tune the GNN model**: Optimize the hyperparameters of the GNN model.
3. **Train the GNN model**: Train the GNN model.
4. **Train the AlphaZero model**: Train the AlphaZero model.
5. **Run a tournament against a random agent**: Conduct a tournament between a specified agent and a random agent.
6. **Exit**: Exit the script.

## Logging

Logging is set up to save all outputs and error messages to a file with a timestamp in its name. This helps in debugging and reviewing the process logs.

## Project Team

- Dag Malstaf (r0799028)
- Pedro Gouveia (r0980130)
- Prateek Grover (r0971804)

---

This project was developed as part of the Machine Learning course [H0T25a] at the Department of Computer Science, KU Leuven. We declare that we are the original and sole authors of this submitted report, except for feedback and support provided by the educational team responsible for guidance.

For more details, refer to our report.

---
