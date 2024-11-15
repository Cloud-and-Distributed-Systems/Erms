# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')  # Set backend to 'Agg' for non-interactive plotting
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import queue
import random

def graphGenerator():
    # generate 100 graph
    graphNum = 100
    
    # Load input data from CSV files
    dfRPC = pd.read_csv('communicationParadigm.csv')  # RPC types and probabilities at different depths
    dfCount = pd.read_csv('depthDistribution.csv')    # Distribution of depth counts
    dfLeaf = pd.read_csv('leafDistribution.csv')      # Probability of leaf nodes at each depth
    dfMedium = pd.read_csv('nodeDistribution.csv')    # Probability of medium nodes at each depth

    # Initialize an empty DataFrame to store generated data
    dfGenerator = pd.DataFrame(columns=['traceid', 'rpcid', 'rpcType', 'depth'])

    def leafMedium(depth):
        """
        Determines if a node is a 'medium' node based on the depth probability.
        Returns True if the node is a medium node.
        """
        leaf_prob = dfLeaf[dfLeaf.depth == depth].percentage.values[0]
        medium_prob = dfMedium[dfMedium.depth == depth].percentage.values[0]
        if random.random() > leaf_prob:
            if random.random() < medium_prob:
                return True
        return False

    def rpcType(depth, num):
        """
        Selects the communication paradigm type (e.g., 'rpc', 'mq', 'db', 'memcached') based on depth and number.
        Uses probability values to choose an communication paradigm type.
        """
        probability = random.random()
        # Fetch probabilities for each RPC type
        rpc = dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'rpc')].percentage.values[0] if not dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'rpc')].empty else 0
        mq = dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'mq')].percentage.values[0] if not dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'mq')].empty else 0
        db = dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'db')].percentage.values[0] if not dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'db')].empty else 0
        memcached = dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'memcached')].percentage.values[0] if not dfRPC[(dfRPC.depth == depth) & (dfRPC.num == num) & (dfRPC.rpc_type == 'memcached')].empty else 0

        # Select RPC type based on cumulative probability
        if probability < rpc:
            return "rpc"
        elif probability < rpc + mq:
            return "mq"
        elif probability < rpc + mq + db:
            return "db"
        elif probability <= rpc + mq + db + memcached:
            return "memcached"
        else:
            return "http"

    def serviceNum(depth):
        """
        Determines the number of services at a specific depth based on cumulative distribution.
        """
        temp = dfCount[dfCount.depth == depth]
        probability = random.random()
        if depth == 1:
            return temp[temp['cumsum'] > probability].num.min() + 1
        else:
            return temp[temp['cumsum'] > probability].num.min()

    # Generate traces
    for traceid in range(graphNum):  # Loop through 100 trace IDs
        q = queue.Queue()
        item = "0"
        rpctype = 'http'
        dfGenerator.loc[len(dfGenerator)] = [traceid, item, rpctype, 1]  # Start trace with root node

        # Check if the root node is a medium node
        if leafMedium(0):
            q.put(item)
        else:
            continue  # Skip this trace if root node is not medium

        # Process nodes in the queue
        while not q.empty():
            item = q.get()
            depth = len(item.split('.'))
            rpcid = 0
            serviceNumTwoTier = serviceNum(depth)  # Determine number of services at current depth
            
            # Generate child nodes for the current node
            for i in range(serviceNumTwoTier):
                itemTemp = f"{item}.{rpcid}"
                rpcid += 1
                rpctype = rpcType(depth, serviceNumTwoTier)  # Select RPC type for child node
                
                # Check if the child node is a medium node, add to queue if so
                if rpctype in ['rpc', 'mq', 'http']:
                    if leafMedium(depth):
                        q.put(itemTemp)

                # Add generated node to DataFrame
                dfGenerator.loc[len(dfGenerator)] = [traceid, itemTemp, rpctype, depth + 1]

    # Save generated trace data to CSV
    dfGenerator.to_csv('generator.csv', index=False)
    
graphGenerator()