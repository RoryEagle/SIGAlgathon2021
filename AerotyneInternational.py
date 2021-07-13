#!/usr/bin/env python

import numpy as np
import math
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.stats import spearmanr
from scipy.stats import entropy
import scipy.optimize

nInst = 100

currentPos = np.zeros(nInst)
globalPrcSoFar = np.zeros((nInst, 600))
returns = np.zeros((nInst, 600))

def getMyPosition (prcSoFar):
    global currentPos

    # Collect global information for optimizer to use
    global globalPrcSoFar
    global returns
    globalPrcSoFar = prcSoFar 
    returns = getReturnsData(prcSoFar)['returns']

    # Determine current position value for reweighting
    positionValues = np.zeros(nInst)

    # Collect the magnitude of each position
    for stock in range(0, currentPos.shape[0] - 1):
        positionValues[stock] = currentPos[stock] * prcSoFar[stock][-1]

    portfolioValue = sum(positionValues)

    portfolioValue = 20000
    alreadyInPosition = False

    for position in currentPos:
        if position != 0: alreadyInPosition = True


    if not alreadyInPosition or prcSoFar.shape[1] % 30 == 0:
        # Initialise a guess using optimal mean-variance weights
        df = pd.DataFrame(prcSoFar.T)
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        ef = EfficientFrontier(mu, S, weight_bounds =(0,1))
        weights = list(ef.max_sharpe().values())
        weights = np.round(weights, 4)
 
        # optimizes weights based on the entropy of the portfolio    
        weights = list(getOptimalAgiusRatioWeights(weights))

        # Collects momentum indicators
        spearmanSignals = generateMASignals(prcSoFar)
    
        for i in range(len(weights)):
            weights[i] = weights[i] * spearmanSignals[i]
        
        # reweights to ensure weights add to 1
        weights = reweight(weights)

        # Gernerates postions from weights
        for stock in range(len(weights)):
            currentPos[stock] = int(math.floor((weights[stock] * portfolioValue) / prcSoFar[stock][-1]))

    return currentPos 


''' Called to optimize the entropy of a portfolio'''
def getOptimalAgiusRatioWeights(initGuess):
    cons = ({'type':'eq', 'fun':check_sum})
    bounds = [(0,1) for _ in range(0,100)]
    assert(len(initGuess) == 100)
    assert(len(bounds) == 100)

    minimizer_kwargs = {'method': 'SLSQP',
                        'bounds':bounds,
                        'constraints': cons}
    opt_results = scipy.optimize.minimize(negativeAgius, initGuess, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':3})
    return opt_results.x

'''Returns the Agius ratio to be optimized for given weights'''
def negativeAgius(weights):
    portfolioReturn = getPortfolioReturns(weights)
    global returns
    partitions = getPartitions()

    entropyValue = calcEntropy(weights, returns, partitions)
    agiusRatio = portfolioReturn / entropyValue

    return entropyValue - 1.7 * portfolioReturn

''' gets the percentage daily retruns of the portfolio'''
def getPortfolioReturns(weights):
    df = pd.DataFrame(globalPrcSoFar.T)
    mu = list(expected_returns.mean_historical_return(df))
    return np.dot(mu , weights)

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return sum(weights) - 1

'''Collects the returns from the whole sample space'''
def getReturnsData(prcSoFar):
    returnsData = {}
    returnsData['returns'] = [[] for k in range(prcSoFar.shape[0])]
    for stock in range(0, prcSoFar.shape[0]):
        returnsData['returns'][stock].append(0)
        for t in range(1, prcSoFar.shape[1]):
            returnsData['returns'][stock].append((prcSoFar[stock][t] - prcSoFar[stock][t - 1])/ prcSoFar[stock][t-1])
    return returnsData


'''Sets the partition sizes for the entropy optimizer'''
def getPartitions():
    low = -0.30
    high = 0.30
    step = 0.0025
    numPartitions = int((high - low) / step)
    p1 = np.linspace(low, high - step, numPartitions)
    p2 = np.linspace(low + step, high, numPartitions)
    p = list(zip(p1,p2))
    return p


'''Calculates the entropy of a given portfolio'''
def calcEntropy(weights, returns, partitions):

    partCounts = [0 for i in range(len(partitions))]

    # Add 1 to partCounts for every observed return instance within relevant partition range
    returns = np.array(returns)
    returns = returns.T
    for t in range(returns.shape[0]):
        portfolioReturn = np.dot(weights, returns[t])
        for i, partition in enumerate(partitions):
            if (portfolioReturn > partition[0] and portfolioReturn <= partition[1]):
                partCounts[i] += 1

    for i in range(len(partCounts)):
        partCounts[i] = partCounts[i] / returns.shape[0]
    return entropy(partCounts)

'''creates MA signals'''
def generateMASignals(prcSoFar):
    windowSizes = [i for i in range(10,110,10)]
    movingAverages = {}
    spearmanSignals = []
    order = [10,9,8,7,6,5,4,3,2,1]

    for stock in range(0, prcSoFar.shape[0]):
        movingAverages[stock] = []
        for size in windowSizes:
            movingAverages[stock].append(np.mean(prcSoFar[stock][-1 * size:]))
        coef, p = spearmanr(movingAverages[stock], order)
        # Set coefficients for multiplication to 1 or 0 depending on momentum factor
        if coef < 0.9: coef = 0
        else: coef = 1
        spearmanSignals.append(round(coef, 4))
    return spearmanSignals

''' reweights the portfolio to have weights sum to one'''
def reweight(weights):
    wsum = sum(weights)
    weights = weights / wsum
    return weights
