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




def cuter_extra_setup_args(cuter_problem):

    equatn = cuter_problem.is_eq_cons
    cu = cuter_problem.cu
    cl = cuter_problem.cl
    bl = cuter_problem.bl
    bu = cuter_problem.bu
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
    setup_args_dict = { 'x': cuter_problem.x0,
                    'bl': bl,
                    'bu': bu,
                    'v': cuter_problem.v0,
                    'cl': cl,
                    'cu': cu,
                    'equatn': equatn,
                    'linear': cuter_problem.is_linear_cons,
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
def v_x(c, adjusted_equatn):
    """
    Calcuate v_x as defined in the paper which is the 1 norm of constraint violation of `c` vector
    :param c: constraint value or linearized constraint value vector
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return:
    """
    if np.any(np.isnan(c)):
        return np.nan
    equality_violation = np.sum(np.abs(c[adjusted_equatn]))
    inequality_violation = np.sum(c[np.logical_and(np.logical_not(adjusted_equatn), (c > 0).flatten())])

    return equality_violation + inequality_violation

def get_phi(x, cuter_problem, setup_args_dict):
    """
    Evaluate merit function phi(x, rho) = f(x) + dist(c(x) | C)
    :param x: current x
    :param rho: penalty parameter
    :param cuter instance
    :return: phi(x, rho) = f(x) + dist(c(x) | C)
    """
    f, _ = cuter_problem.obj(x, gradient = False)
    c, _ = cuter_problem.cons(x, gradient=False)

    return v_x(c, setup_args_dict['adjusted_equatn']) + f

def get_delta_phi(x0, x1,cuter_problem, setup_args_dict):
    """
    Evaluate delta merit function phi(x0, rho) - phi(x1, rho)
	"""
    return get_phi(x0, cuter_problem, setup_args_dict) - get_phi(x1, cuter_problem, setup_args_dict)

def linearModelPenalty(A, b, g, d, adjusted_equatn):
    """
    Calculate the l(d, rho; x) defined in the paper which is the linearized model of penalty function
    f(x) + dist(c(x) | C)
    :param A: Jacobian of constraints
    :param b: c(x) constraint function value
    :param g: gradient of objective
    :param rho: penalty parameter
    :param d: current direction
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return: l(d ; x)
    """

    c = A.dot(d) + b
    linear_model = g.T.dot(d) + v_x(c, adjusted_equatn)
    return linear_model[0, 0]

def l0(b, equatn):
    # Line 201
    b = b.reshape(1, -1)[0]
    return np.sum(np.abs(b[equatn == True])) + np.sum(b[np.logical_and(equatn == False, b>0)])

def getPrimalObject(primal_var, g, equatn):
    # Line 757, formula 4.1
    return primal_var.dot(makeC(g, equatn));
def getDualObject(A, g, b, dual_var, delta):
    # Line 763, formula 4.2
    return  \
        (b.T.dot(dual_var) -  \
        delta * np.sum(np.abs(g.T + dual_var.T.dot(A))))[0]
def getRatioC(A, b, dual_var, primal_var, equatn, l_0):
    # line 227: r_c = 1 - sqrt(X/l_0)
    # line 221: X = sum((1-dual(i)) * v(<a, d> + b)) + sum((1+dual(j)) * v(<a, d> + b))
    #           where i in E+(d), I+(d), j in I-(d) 
    X = 0
    m, n = A.shape
    for i in range(m):
        x_new = A[i, :].dot(primal_var) + b[i, 0]
        if x_new > 0:
            X += (1-dual_var[i]) * x_new
        elif x_new < 0 and equatn[i] == True:            
            X += (1+dual_var[i]) * np.abs(x_new)
    return 1-np.sqrt(np.abs(X / (l_0 + 1e-8)))
def getRatio(A, b, g,  primal_var, dual_var, delta, equatn, l_0):
    # Line 199, formula 2.16.
    # When rho is set to 0, it calculates ratio_fea, or it calculates ratio_obj
    primal_obj = getPrimalObject(primal_var, g,  equatn)
    dual_obj = getDualObject(A, g,  b, dual_var, delta)
    
    up = l_0 - primal_obj
    down = (l_0 - dual_obj)
    down += 1e-5

    return up/down








def getLinearSearchDirection(A, b, g, delta, cuter_problem, setup_args_dict, dust_param):
    equatn = setup_args_dict['adjusted_equatn']
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    m, n = A.shape
    c_, A_, b_, basis_ = makeC(g, equatn), makeA(A), makeB(b, delta, n), makeBasis(b, n)
    
    # Construct a simplex problem instance.
    linsov = Simplex(c_, A_, b_, basis_)
    primal = linsov.getPrimalVar()

    beta_fea = dust_param.beta_fea 
    beta_opt = dust_param.beta_opt
    theta = dust_param.theta
    
    dual_var = np.zeros(m)
    primal_var = np.zeros(n)
    ratio_opt = 1;
    ratio_fea = 1;
    ratio_c = 0

    l_0 = l0(b, equatn)
    iter_cnt = 0
    while not linsov.isOptimal():
        iter_cnt += 1;
        if iter_cnt > dust_param.max_sub_iter:
            break;
        # update the basis.
        linsov.updateBasis()

        # primal has size 4*n + 2*m
        # but we are only interested with the first 2n, as they are plus and minus of d.
        primal = linsov.getPrimalVar()
        # primal_var = d+ - d-
        primal_var = primal[0:n] - primal[n:2*n]

        # dual_var also has size m+2n,
        # but we are only interested with the first m.
        dual = -linsov.getDualVar()
        dual_var = dual[0, 0:m]
        # Truncation in-equality constraint to [0, 1], equality constraint to [-1, 1]
        dual_var = np.maximum(np.minimum(dual_var, np.ones([m])), -1 * equatn)
        # Update ratios.
        ratio_opt = getRatio(A, b, g, primal, dual_var, delta, equatn, l_0)
        ratio_c = getRatioC(A, b, dual_var, primal_var, equatn, l_0)

        if ratio_c >= beta_fea and ratio_opt >= beta_opt and ratio_fea >= beta_fea:
        # Should all satisfies, break.
            break
        elif ratio_c >= beta_fea and ratio_opt >= beta_opt:
        # Update rho if needed.
            linsov.resetC(makeC(g, equatn))

def get_f_g_A_b_violation(x_k, cuter_problem, setup_args_dict):
    """
    Calculate objective function, gradient, constraint function and constraint Jacobian
    :param x_k: current iteration x
    :param cuter instance
    :param dust_param: dust param instance
    :return:
    """
    f, g = cuter_problem.obj(x_k, gradient = True)
    b, A = cuter_problem.cons(x_k, gradient = True)
    violation = v_x(b, setup_args_dict['adjusted_equatn'])

    return f, g, b, A, violation














def linearSolveTrustRegion(cuter_problem, dust_param,setup_args_dict):
    """
    Non linear solver for cuter problems
    :param cuter instance
    :param dust_param: dust parameter class instance
    :param logger: logger instance to store log information
    :return:
        status:
                -1 - max iteration reached
                1 - solve the problem to optimality
    """
    def get_KKT(A, b, g, eta):
        """
        Calcuate KKT error
        :param A: Jacobian of constraints
        :param b: c(x) constraint function value
        :param g: gradient of objective
        :param eta: multiplier in canonical form dual problem
        :return: kkt error
        """

        err_grad = np.max(np.abs(A.T.dot(eta) + g))    
        err_complement = np.max(np.abs(eta * b))
        return max(err_grad, err_complement)

    x_0 = setup_args_dict['x']
    # num_var = setup_args_dict['n'][0]
    # beta_l = 0.6 * dust_param.beta_opt * (1 - dust_param.beta_fea)
    adjusted_equatn = setup_args_dict['adjusted_equatn']
    zero_d = np.zeros(x_0.shape)
    i, status = 0, -1
    x_k = x_0.copy()

    f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter_problem, setup_args_dict)
    # m, n = A.shape
    max_iter = dust_param.max_iter
    #init_kkt = get_KKT(A, b, g, np.zeros((m, 1)), rho)

    all_kkt_erros, all_violations, all_fs, all_sub_iter = \
    [1], [violation], [f], []

    delta = dust_param.init_delta; 
    
    while i < max_iter:

        # DUST / PSST / Subproblem here.
        d_k, dual_var, ratio_complementary, ratio_opt, ratio_fea, sub_iter = \
            getLinearSearchDirection(A, b, g, delta, cuter_problem, setup_args_dict,dust_param)

        # 2.3
        l_0_x_k = linearModelPenalty(A, b, g, zero_d, adjusted_equatn)
        l_d_x_k = linearModelPenalty(A, b, g, d_k, adjusted_equatn)
        delta_linearized_model = l_0_x_k - l_d_x_k

        
        kkt_error_k = get_KKT(A, b, g, dual_var)
        init_kkt = max(kkt_error_k, 1)
        

        # Update delta.
        if ratio_opt > 0:
            sigma = get_delta_phi(x_k, x_k+d_k, cuter_problem, setup_args_dict) / (delta_linearized_model + 1e-5)
            if np.isnan(sigma):
                # Set it to a very small value to escape inf case.
                sigma = -0x80000000
            if sigma < dust_param.SIGMA:
                delta = max(0.5*delta, dust_param.MIN_delta)
            elif sigma > dust_param.DELTA:
                delta = min(2*delta, dust_param.MAX_delta)

        # ratio_opt: 3.6. It's actually r_v in paper.


        f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter_problem, dust_param)
        kkt_error_k = get_KKT(A, b, g, dual_var) / init_kkt

        # Store iteration information
        # all_violations.append(violation)
        # all_fs.append(f)
        # all_kkt_erros.append(kkt_error_k)
        # all_sub_iter.append(sub_iter)

        if kkt_error_k < dust_param.eps_opt and violation < dust_param.eps_violation:
            status = 1
            break
        i += 1
        # logger.info('-' * 200)
        
    # return {'x': x_k, 'dual_var': dual_var,  'status': status, 'obj_f': f, 'x0': setup_args_dict['x'],'kkt_error': kkt_error_k, 'iter_num': i, 'constraint_violation': violation, 'violations': all_violations, 'fs': all_fs, 'subiters': all_sub_iter, 'kkt_erros': all_kkt_erros, 'num_var': num_var, 'num_constr': dual_var.shape[0], 'pivot_cnt': pivot_cnt}
