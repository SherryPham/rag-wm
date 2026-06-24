#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical utilities for watermark detection.
"""

import math
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp


def f1_score(precision, recall):
    """
    Calculate F1 score given precision and recall arrays.
    
    Args:
        precision: Precision values (array)
        recall: Recall values (array)
    
    Returns:
        np.array: F1 scores
    """
    f1_scores = np.divide(
        2 * precision * recall, 
        precision + recall, 
        where=(precision + recall) != 0
    )
    return f1_scores


def binomial_test_greater(x: int, n: int, p0: float) -> float:
    """
    Exact one-sided binomial test: p-value = P[X >= x | n, p0]
    
    Args:
        x: Number of successes
        n: Total trials
        p0: Null hypothesis probability
        
    Returns:
        float: p-value
    """
    p_val = 0.0
    for k in range(x, n + 1):
        p_val += math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    return min(1.0, max(0.0, p_val))



def calculate_neg_log_p(n, k, p0):
    # 1. Identify the range of the "tail" (from k to n)
    #    In your case: 229, 230, ..., 250
    tail_range = np.arange(k, n + 1)
    
    # 2. Calculate the Log PMF for each value in the tail
    #    This gives us [log(P(229)), log(P(230)), ...]
    #    These values are safe (e.g., -980.5) and won't underflow.
    log_pmfs = binom.logpmf(tail_range, n, p0)
    
    # 3. Sum them safely in log-space
    #    This performs log(exp(p1) + exp(p2) + ...) without leaving log-space
    log_p_value_natural = logsumexp(log_pmfs)
    
    # 4. Convert to base-10 scale for readability
    neg_log10_p = -(log_p_value_natural / np.log(10))
    
    return neg_log10_p


