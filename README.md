# Genetic Programming for Time Series Forecasting

This project applies a tree-based genetic programming algorithm to the problem of time series forecasting. The project was submitted as part of my MSc Advanced Computer Science, and was required to perform three functions:

1. Parse mathematical expressions.
2. Calculate the fitness of a mathematical expression in forecasting a time series dataset.
3. Generate mathematical expressions which forecast a time series dataset.

The project achieved a grade of 96%. An explanation of the implemented algorithm, as well as analysis of results, can be found in the [Report.pdf](Report.pdf) file.

## Prerequisites

The entire program was built using base Python 3.8. The project is only guaranteed to work with Python versions 3.8 or higher, however, it should also work with earlier Python versions. Docker is also required to run tests, but this is optional.

## How To Use

This project is built to be run from the console, and can be done so by using the following command:

```
python Genetic_Programming.py [flags]
```

The program requires the use of flags to operate correctly. Listed below are the accepted flags:

- `-expr` : A mathematical expression.
- `-x` : An input vector of data points.
- `-n` : The dimension of the input vector.
- `-data` : The name of a file containing the training data in the form of m lines, where each line contains n + 1 values separated by tab characters. The first n elements in a line represent an input vector x, and the last element in a line represents the output value y.
- `-m` : The size of the training data.
- `-lambda` : Population size.
- `-time_budget` : The number of seconds to run the algorithm.
- `-question` : Takes a value of 1-3 and indicates the operation to perform.

Due to the way the coursework was structured. The project can perform one of three operations, with  the required operation being signalled by the `-question` flag. Listed below is a description of each operation and its required flags:

- `-question 1` : Parses and evaluates a given expression. Requires the flags `expr`, `x`, and `n`.
- `-question 2` : Computes the fitness of an expression given a training set. Requires the flags `expr`, `n`, `m`, and `data`.
- `-question 3` : Runs the genetic programming algorithm to produce a mathematical expression for the given time series training set. Requires the flags `lambda`, `n`, `m`, `data`, and `time_budget`.

### Testing

The simplest way to run the test suite is through the use of Docker. Simply build, and run, the provided Dockerfile.

If problems are encountered, a random assortment of tests has also been provided as shell scripts. These can be found in the [Tests](Tests/) directory. Each files name corresponds to the operation it is testing.

## License

This project is licensed under the terms of the [Creative Commons Attribution 4.0 International Public license](License.md).
