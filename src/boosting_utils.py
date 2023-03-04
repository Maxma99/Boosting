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






def cuter_extra_setup_args(cuter_problem):

    equatn = cuter_problem.is_eq_cons()
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
                    'linear': cuter_problem.is_linear_cons(),
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
    :param rescale: if true, solve the rescale problem
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
    rho*f(x) + dist(c(x) | C)
    :param A: Jacobian of constraints
    :param b: c(x) constraint function value
    :param g: gradient of objective
    :param rho: penalty parameter
    :param d: current direction
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return: l(d, rho; x)
    """

    c = A.dot(d) + b
    linear_model = g.T.dot(d) + v_x(c, adjusted_equatn)
    return linear_model[0, 0]

def l0(b, equatn, omega):
    # Line 201
    b = b.reshape(1, -1)[0]
    return np.sum(np.abs(b[equatn == True])) + np.sum(b[np.logical_and(equatn == False, b>0)]) + omega

def getLinearSearchDirection(A, b, g, delta, cuter_problem, setup_args_dict, dust_param, omega):
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

    l_0 = l0(b, equatn, omega)
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
        nu_var = -linsov.getNuVar(makeC(g*0, equatn))

        # Update ratios.
        ratio_fea = getRatio(A, b, g, 0, primal, nu_var[0:m], delta, equatn, l_0)
        ratio_opt = getRatio(A, b, g, rho, primal, dual_var, delta, equatn, l_0)
        ratio_c = getRatioC(A, b, dual_var, primal_var, equatn, l_0)

        if ratio_c >= beta_fea and ratio_opt >= beta_opt and ratio_fea >= beta_fea:
        # Should all satisfies, break.
            break
        elif ratio_c >= beta_fea and ratio_opt >= beta_opt:
        # Update rho if needed.
            rho *= theta
            linsov.resetC(makeC(g*rho, equatn))

def get_f_g_A_b_violation(x_k, cuter_problem, dust_param, setup_args_dict):
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














def linearSolveTrustRegion(cuter_problem, dust_param,setup_args_dict, logger):
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
    num_var = setup_args_dict['n'][0]
    beta_l = 0.6 * dust_param.beta_opt * (1 - dust_param.beta_fea)
    adjusted_equatn = setup_args_dict['adjusted_equatn']
    zero_d = np.zeros(x_0.shape)
    i, status = 0, -1
    x_k = x_0.copy()

    logger.info('-' * 200)
    logger.info(
        '''{0:4s} | {1:13s} | {2:12s} | {3:12s} | {4:12s} | {5:12s} | {6:12s} | {7:12s} | {8:12s} | {9:12s} | {10:6s} | {11:12s} | {12:12s} | {13:12s}'''.format(
            'Itr', 'KKT', 'Delta', 'Violation', 'Rho', 'Objective', 'Ratio_C', 'Ratio_Fea', 'Ratio_Opt', 'step_size',
            'SubItr', 'Delta_L', 'Merit', "||d||"))

    f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter_problem, dust_param)
    m, n = A.shape
    rho = 1
    omega = dust_param.init_omega
    max_iter = dust_param.max_iter
    rescale = False
    #init_kkt = get_KKT(A, b, g, np.zeros((m, 1)), rho)

    all_kkt_erros, all_violations, all_fs, all_sub_iter = \
    [1], [violation], [f], []

    delta = dust_param.init_delta; 
    step_size = 1;
    logger.info(
        '''{0:4d} |  {1:+.5e} | {2:+.5e} | {3:+.5e} | {4:+.5e} |  {6:+.5e} | {7:+.5e} | {8:+.5e} | {9:+.5e} | {10:6d} | {11:+.5e} | {12:+.5e} | {13:+.5e}''' \
            .format(i, 1, delta, violation,  f, -1, -1, -1, step_size, -1, -1, f + violation, -1))
    
    fn_eval_cnt = 0
    pivot_cnt = 0
    while i < max_iter:

        # DUST / PSST / Subproblem here.
        d_k, dual_var, rho, ratio_complementary, ratio_opt, ratio_fea, sub_iter = \
            getLinearSearchDirection(A, b, g, rho, delta, cuter_problem, dust_param, omega)

        # 2.3
        l_0_rho_x_k = linearModelPenalty(A, b, g, rho, zero_d, adjusted_equatn)
        l_d_rho_x_k = linearModelPenalty(A, b, g, rho, d_k, adjusted_equatn)
        delta_linearized_model = l_0_rho_x_k - l_d_rho_x_k

        # 2.2
        l_0_0_x_k = linearModelPenalty(A, b, g, 0, zero_d, adjusted_equatn)
        l_d_0_x_k = linearModelPenalty(A, b, g, 0, d_k, adjusted_equatn)
        delta_linearized_model_0 = l_0_0_x_k - l_d_0_x_k
        
        kkt_error_k = get_KKT(A, b, g, dual_var, rho)

        # Relative KKT
        if i == 0:
            init_kkt = max(kkt_error_k, 1)
            if kkt_error_k < dust_param.eps_opt and violation < dust_param.eps_violation:
                status = 1
                break
        else:
            kkt_error_k /= init_kkt

        # Update delta.
        if ratio_opt > 0:
            sigma = get_delta_phi(x_k, x_k+d_k, rho, cuter_problem, rescale, delta) / (delta_linearized_model + 1e-5)
            if np.isnan(sigma):
                # Set it to a very small value to escape inf case.
                sigma = -0x80000000
            if sigma < dust_param.SIGMA:
                delta = max(0.5*delta, dust_param.MIN_delta)
            elif sigma > dust_param.DELTA:
                delta = min(2*delta, dust_param.MAX_delta)

        # ratio_opt: 3.6. It's actually r_v in paper.
        if ratio_opt > 0:
            step_size = line_search_merit(x_k, d_k, rho, delta_linearized_model, dust_param.line_theta, cuter_problem,
                                          dust_param.rescale)
            x_k += d_k * step_size
            fn_eval_cnt += 1 - np.log2(step_size)
        else:
            fn_eval_cnt += 1

        # PSST
        if delta_linearized_model_0 > 0 and \
                delta_linearized_model + omega < beta_l * (delta_linearized_model_0 + omega):
            # TODO: Change this update, as we are using linear.
            rho = (1 - beta_l) * (delta_linearized_model_0 + omega) / (g.T.dot(d_k))[0, 0]

        f, g, b, A, violation = get_f_g_A_b_violation(x_k, cuter_problem, dust_param)
        kkt_error_k = get_KKT(A, b, g, dual_var, rho) / init_kkt

        omega *= dust_param.omega_shrink

        # Store iteration information
        all_violations.append(violation)
        all_fs.append(f)
        all_kkt_erros.append(kkt_error_k)
        all_sub_iter.append(sub_iter)

        pivot_cnt += sub_iter

        logger.info(
            '''{0:4d} |  {1:+.5e} | {2:+.5e} | {3:+.5e} | {4:+.5e} | {5:+.5e} | {6:+.5e} | {7:+.5e} | {8:+.5e} | {9:+.5e} | {10:6d} | {11:+.5e} | {12:+.5e} | {13:+.5e}''' \
                .format(i, kkt_error_k, delta, violation, rho, f, ratio_complementary, ratio_fea, ratio_opt, step_size,
                        sub_iter, delta_linearized_model, rho * f + violation, np.linalg.norm(d_k, 2)))

        if kkt_error_k < dust_param.eps_opt and violation < dust_param.eps_violation:
            status = 1
            break
        i += 1
        if (np.linalg.norm(d_k, 2) < 1e-10):
            rho *= dust_param.theta
    logger.info('-' * 200)

    if rescale:
        f = f / setup_args_dict['obj_scale']

    return {'x': x_k, 'dual_var': dual_var, 'rho': rho, 'status': status, 'obj_f': f, 'x0': setup_args_dict['x'],
            'kkt_error': kkt_error_k, 'iter_num': i, 'constraint_violation': violation, 'rhos': all_rhos,
            'violations': all_violations, 'fs': all_fs, 'subiters': all_sub_iter, 'kkt_erros': all_kkt_erros,
            'fn_eval_cnt': fn_eval_cnt, 'num_var': num_var, 'num_constr': dual_var.shape[0], 'pivot_cnt': pivot_cnt}
