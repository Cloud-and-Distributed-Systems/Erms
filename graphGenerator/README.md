# Microservice Graph Generator

This microservice graph generator simulates a hierarchical structure of microservices based on probability distributions from input CSV files. It generates a graph-like structure where each node represents a microservice call, and each edge represents an invocation from one service to another.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Input Files](#input-files)
- [Running the Generator](#running-the-generator)
- [Output](#output)
- [Explanation of Functions](#explanation-of-functions)
- [Customization](#customization)
- [License](#license)

## Overview

The generator produces microservice trace data by reading from four input files, each detailing a specific probability distribution. This data simulates realistic microservice interactions and includes different invocation types (e.g., RPC, MQ, DB). The generated output can be used for studying microservice behavior, testing distributed systems, or benchmarking resource management tools.

## Prerequisites

- **Python 3.x**
- Required Python libraries:
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `scipy`
  - `queue`

Install the necessary libraries with:
```bash
pip3 install matplotlib numpy pandas scipy


python3 microservice_graph_generator.py

