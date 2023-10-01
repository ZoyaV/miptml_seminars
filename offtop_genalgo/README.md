

# Genetic Algorithms (GA) Guide


## Table of Contents

- [Introduction to GA](https://github.com/ZoyaV/miptml_seminars/blob/main/offtop_genalgo/Intro.ipynb)
- [Strategies in GA](https://github.com/ZoyaV/miptml_seminars/blob/main/offtop_genalgo/Genetic_Algorithms_Strategies.ipynb)

## Introduction

Genetic algorithms are a type of optimization and search algorithm inspired by the process of natural selection. They are used to find approximate solutions to optimization and search problems.

## Theory

- **Chromosomes**: A solution in the problem space.
- **Population**: A set of possible solutions (chromosomes).
- **Generations**: Iterations in the algorithm where populations evolve.
- **Selection**: Process by which chromosomes are chosen for reproduction based on their fitness.
- **Crossover (Recombination)**: Combining two chromosomes to produce a new chromosome.
- **Mutation**: Introducing small random changes in a chromosome.
- **Fitness Function**: A function that tells how close a given solution is to the optimum solution.

## When to Use

1. When the search space is large, complex, or poorly understood.
2. When the exact optimization algorithm for the problem is not known.
3. For combinatorial problems where traditional methods are inefficient.

## Strategies in Genetic Algorithms

1. **Selection Strategies**:
   - Fitness Proportionate Selection
   - Tournament Selection
   - Rank Selection
   - Elitism

2. **Crossover Strategies**:
   - One-point Crossover
   - Two-point Crossover
   - Uniform Crossover

3. **Mutation Strategies**:
   - Bit Flip Mutation
   - Swap Mutation
   - Inversion Mutation

## DEAP Library

[DEAP (Distributed Evolutionary Algorithms in Python)](https://deap.readthedocs.io/en/master/) is a popular Python library for evolutionary algorithms.

- **Installation**: `pip install deap`
- **Features**:
  - Provides tools to design custom evolutionary algorithms.
  - Supports parallelization of evaluations.
  - Offers benchmarks and tools to analyze and visualize results.

### Basic Usage:

```python
from deap import base, creator, tools

# Define the fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register functions
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define evaluation, crossover, mutation, and selection
toolbox.register("evaluate", your_evaluation_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Example of creating a population and evaluating it
population = toolbox.population(n=300)
fitnesses = list(map(toolbox.evaluate, population))
```

