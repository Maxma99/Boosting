import pycutest as pc
import numpy as np
from solver_util import makeA, makeB, makeC, makeBasis
from simplex import Simplex
import autograd.numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from autograd import grad
from collections import Counter, defaultdict
from itertools import compress
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds
from scipy.special import huber
from sklearn.preprocessing import StandardScaler
from time import process_time
from linear_solver import linearSolveTrustRegion
from param import DustParam
import warnings
warnings.simplefilter('ignore')

STEP_SIZE_MIN = 1e-10
__UP = 1e20
__LOW = -1e20
__PMAX = 1e3


def segment_search(f, grad_f, x, y, tol=1e-6, stepsize=True):
    
    '''
    Minimizes f over [x, y], i.e., f(x+gamma*(y-x)) as a function of scalar gamma in [0,1]
    '''
    
    # restrict segment of search to [x, y]
    d = (y-x).copy()
    left, right = x.copy(), y.copy()
    
    # if the minimum is at an endpoint
    if np.dot(d, grad_f(x))*np.dot(d, grad_f(y)) >= 0:
        if f(y) <= f(x):
            return y, 1
        else:
            return x, 0
    
    # apply golden-section method to segment
    gold = (1+np.sqrt(5))/2
    improv = np.inf
    while improv > tol:
        old_left, old_right = left, right
        new = left+(right-left)/(1+gold)
        probe = new+(right-new)/2
        if f(probe) <= f(new):
            left, right = new, right
        else:
            left, right = left, probe
        improv = np.linalg.norm(f(right)-f(old_right))+np.linalg.norm(f(left)-f(old_left))
    x_min = (left+right)/2
    
    # compute step size gamma
    gamma = 0
    if stepsize == True:
        for i in range(len(d)):
            if d[i] != 0:
                gamma = (x_min[i]-x[i])/d[i]
                break
                
    return x_min, gamma

def sigd(f, grad_f, x, S, alpha):
    
    '''
    Simplex Gradient Descent
    '''
    
    g = np.dot(S, grad_f(x))
    e, k = np.ones(len(S)), len(S)
    d = g-np.dot(g, e)*e/k
    
    if np.linalg.norm(d) == 0:
        return S[0], [S[0]], [1]
    
    eta = np.min([alpha[i]/d[i] if d[i] > 0 else np.inf for i in range(len(alpha))])
    beta = np.array(alpha)-eta*d
    y = np.dot(beta, S)

    if f(x) >= f(y):
        idx = list(beta > 0)
        return y, list(compress(S, idx)), list(compress(beta, idx))
    
    else:
        x, gamma = segment_search(f, grad_f, x, y)
        return x, S, list((1-gamma)*np.array(alpha)+gamma*beta)

def find_index(v, S):
    
    for i in range(len(S)):
        if np.all(S[i] == v):
            return i
        
    return -1

def align(d, hat_d):
    
    if np.linalg.norm(hat_d) < 1e-15:
        return -1
    
    else:
        return np.dot(d, hat_d)/(np.linalg.norm(d)*np.linalg.norm(hat_d))

def nnmp(x, grad_f_x, align_tol, K, traffic):
    
    '''
    Minimizes ||-grad_f_x-d||_2^2/2 s.t. d in cone(V-x)
    '''
    
    d, Lbd, flag = np.zeros(len(x)), 0, True
    
    G = grad_f_x+d
    align_d = align(-grad_f_x, d)
    
    for k in range(K):
        
        if traffic == True:
            G = np.maximum(G, 0)
        
        u = lmo(G)-x
        d_norm = np.linalg.norm(d)
        if d_norm > 0 and np.dot(G, -d/d_norm) < np.dot(G, u):
            u = -d/d_norm
            flag = False
        lbd = -np.dot(G, u)/np.linalg.norm(u)**2
        dd = d+lbd*u
        align_dd = align(-grad_f_x, dd)
        align_improv = align_dd-align_d
        
        if align_improv > align_tol:
            d = dd
            Lbd = Lbd+lbd if flag == True else Lbd*(1-lbd/d_norm)
            G = grad_f_x+d
            align_d = align_dd
            flag = True
            
        else:
            break
        
    return d/Lbd, k, align_d

def boostfw(f, grad_f, L, x, step='ls', f_tol=1e-6, time_tol=np.inf, align_tol=1e-3, K=500, traffic=False):
    
    if traffic == False:
        values, times, oracles, gaps = [f(x)], [0], [0], [np.dot(grad_f(x), x-lmo(grad_f(x)))]
        f_improv = np.inf

        start = process_time()
        x = lmo(grad_f(x))
        end = process_time()
        values.append(f(x))
        times.append(end-start)
        oracles.append(1)
    else:
        values, times, oracles, gaps = [f(x)], [0], [0], []
        f_improv = np.inf
    
    while f_improv > f_tol and np.sum(times) < time_tol:
                
        f_old = f(x)
        start = process_time()
        
        grad_f_x = grad_f(x)
        
        t1 = process_time()
        gaps.append(np.dot(grad_f_x, x-lmo(grad_f_x)))
        t2 = process_time()
        
        g, num_oracles, align_g = nnmp(x, grad_f_x, align_tol, K, traffic)
        
        if step == 'L':
            gamma = min(align_g*np.linalg.norm(grad_f_x)/(L*np.linalg.norm(g)), 1)
            x = x+gamma*g
        elif step == 'ls':
            x, gamma = segment_search(f, grad_f, x, x+g)
        else:
            gamma = min(-np.dot(g, grad_f_x)/np.dot(g, (np.dot(step, g))), 1)
            x = x+gamma*g
        
        end = process_time()
        values.append(f(x))
        times.append(end-start+t1-t2)
        oracles.append(num_oracles)
        f_improv = f_old-f(x)
        
    return x, values, times, oracles, gaps






def cuter_extra_setup_args(problem):

    equatn = problem.is_eq_cons()
    cu = problem.cu
    cl = problem.cl
    bl = problem.bl
    bu = problem.bu
    iequatn = np.logical_not(equatn)
    inequality_upper = np.logical_and((cu != __UP).flatten(), iequatn)
    inequality_lower = np.logical_and((cl != __LOW).flatten(), iequatn)
    is_lower_bound_only_constr = np.logical_and(inequality_lower, np.logical_not(inequality_upper))
    is_double_bound_constr = np.logical_and(inequality_lower, inequality_upper)
    bl_flag = (bl != __LOW).flatten()
    bu_flag = (bu != __UP).flatten()
    num_added_ineq_constr = np.sum(is_double_bound_constr) + np.sum(bl_flag) + np.sum(bu_flag)
    added_iequatn = np.array([False] * num_added_ineq_constr, dtype=bool)
    adjusted_equatn = np.hstack([equatn, added_iequatn])
    setup_args_dict = { 'x': problem.x0,
                    'bl': bl,
                    'bu': bu,
                    'v': problem.v0,
                    'cl': cl,
                    'cu': cu,
                    'equatn': equatn,
                    'linear': problem.is_linear_cons(),
                    'bl_flag': bl_flag,
                    'bu_flag': bu_flag,
                    'adjusted_equatn': adjusted_equatn,
                    'iequatn': iequatn,
                    'inequality_lower': inequality_lower,
                    'inequality_upper': inequality_upper,
                    'is_lower_bound_only_constr': is_lower_bound_only_constr,
                    'is_double_bound_constr': is_double_bound_constr}
    return setup_args_dict

# No rescaling

