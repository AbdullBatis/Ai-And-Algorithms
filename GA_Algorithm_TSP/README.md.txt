# Traveling Salesman Problem (TSP) using Genetic Algorithm

## Overview
This project aims to solve the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA). The problem involves finding the shortest possible route that visits a set of cities and returns to the origin city. This implementation leverages various genetic algorithm techniques, such as crossover, mutation, and selection, to evolve a population of potential solutions towards the optimal route.

## Problem Statement
The Traveling Salesman Problem is a classic optimization problem where the goal is to determine the most efficient route to visit a set of cities. The challenge lies in the fact that as the number of cities increases, the number of possible routes grows exponentially, making it computationally expensive to solve using brute force.

## Features
- Genetic Algorithm-based optimization for solving TSP.
- Visualization of city locations and the resulting optimal route.
- Multiple experiments to test the effect of different parameters (e.g., number of generations, population size).
- **Results**: Visualization of fitness progress over generations (minimum and average fitness).
- **Coordinate Data**: The dataset for 200 cities is provided for use.

## How It Works
1. **Initialization**:
   - The cities' coordinates are loaded from a `.tsp` file.
   - The distance matrix between every pair of cities is computed using the Euclidean distance formula.

2. **Genetic Algorithm**:
   - **Population Initialization**: A population of randomly shuffled cities is created.
   - **Selection**: Tournament selection is used to select individuals for reproduction.
   - **Crossover**: Ordered crossover (OX) is applied to produce offspring by combining two parent solutions.
   - **Mutation**: Random mutations are applied to shuffle the order of cities in an individual solution.
   - **Fitness Evaluation**: The fitness of each individual is evaluated by calculating the total route distance.
   - **Termination**: The algorithm runs for a predefined number of generations.

3. **Visualization**:
   - The cities are plotted using `matplotlib`, with the optimal route visualized as a line connecting the cities.

## Requirements
- Python 3
- Libraries:
    - numpy
    - matplotlib
    - seaborn
    - deap (for genetic algorithm operations)

## Open Source and Improvements
The initial code for this project is based on open-source implementations of the Traveling Salesman Problem using Genetic Algorithms. Through experimentation and improvements, I was able to achieve the best fitness with optimized parameters. The current parameter settings have shown to provide the best results for this problem, and the algorithm is performing at its maximum fitness under these conditions.
