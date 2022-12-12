# Maze

The environment is a square maze, with walls along the way. The Agent starts in the top left of the maze, and has to find its way to the bottom right.

Example with a maze of size 20*20, with the Q-Learning algorithm:

First Episodes | Episode 10 | Episodes 45-50
:-------------------------:|:-------------------------:|:-------------------------:
![2022-12-11_22-46-22](https://user-images.githubusercontent.com/64380881/206930804-530f318a-2183-4ee7-9bdc-4b969fe4a5b9.gif) | ![2022-12-11_22-46-01](https://user-images.githubusercontent.com/64380881/206931024-7fe8c6c1-3168-4156-abf6-e24e7db6b68c.gif) |  ![2022-12-11_22-47-01](https://user-images.githubusercontent.com/64380881/206931034-8963612b-2bf6-4d40-8d4a-c974d4a43f0d.gif)

# Repo structure

### [algos](algos/)

Different algorithms

### [maze_solving.ipynb](maze_solving.ipynb)

Notebook where each algorithm is run, with additional visualizations

### [run.py](run.py)

Python file to execute in order to run the code

# Example of use

The run script takes several required arguments:
- `size`: size of the maze
- `algo`: algorithm to use

In order to run the script using Q-Learning in a maze of shape 10*10, run:

```
python run.py --size 10 --algo QLearning
```
