import pycutest as pc
import numpy as np
from solver_util import makeA, makeB, makeC, makeBasis
import autograd.numpy as np
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
                    'is_double_bound_constr': is_double_bound_constr,
                    "init_rho": 1,
                    "init_omega": 1e-2,
                    "init_delta": 1,
                    "max_iter": 512,
                    "max_sub_iter": 512,
                    "beta_opt": 0.75,
                    "beta_fea": 0.3,
                    "theta": 0.9,
                    "line_theta": 1e-4,
                    "omega_shrink": 0.7,
                    "eps_opt": 1e-4,
                    "eps_violation": 1e-3,
                    "sub_verbose": False,
                    "rescale": False,
                    "SIGMA": 0.3,
                    "DELTA": 0.75,
                    "MIN_delta": 1e-2,
                    "MAX_delta": 64}
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
    Evaluate merit function phi(x, rho) = f + dist(c(x) | C)
    :param x: current x
    :param rho: penalty parameter
    :param cuter instance
    :return: phi(x, rho) = f + dist(c(x) | C)
    """
    f, _ = cuter_problem.obj(x, gradient = False)
    c, _ = cuter_problem.cons(x, gradient=False)

    return v_x(c, setup_args_dict['equatn']) + f

def get_delta_phi(x0, x1,cuter_problem, setup_args_dict):
    """
    Evaluate delta merit function phi(x0, rho) - phi(x1, rho)
	"""
    return get_phi(x0, cuter_problem, setup_args_dict) - get_phi(x1, cuter_problem, setup_args_dict)

def linearModelPenalty(grad_c, c, grad_f, d, adjusted_equatn):
    """
    Calculate the l(d, rho; x) defined in the paper which is the linearized model of penalty function
    f + dist(c(x) | C)
    :param grad_c: Jacobian of constraints
    :param c: c(x) constraint function value
    :param grad_f: gradient of objective
    :param d: current direction
    :param adjusted_equatn: boolean vector indicating if it is equation constraint
    :return: l(d ; x)
    """
    c_gc = grad_c.dot(d) + c
    linear_model = grad_f.T.dot(d) + v_x(c_gc, adjusted_equatn)
    subg_model = grad_f.T.dot(d) + v_x(grad_c.dot(d), adjusted_equatn)
    return linear_model[0, 0], subg_model[0, 0]

def l0(c, equatn):
    # Line 201
    c = c.reshape(1, -1)[0]
    return np.sum(np.abs(c[equatn == True])) + np.sum(c[np.logical_and(equatn == False, c>0)])

def getPrimalObject(primal_var, grad_f, equatn):
    # Line 757, formula 4.1
    return primal_var.dot(makeC(grad_f, equatn));
def getDualObject(grad_c, grad_f, c, dual_var, delta):
    # Line 763, formula 4.2
    return  \
        (c.T.dot(dual_var) -  \
        delta * np.sum(np.abs(grad_f.T + dual_var.T.dot(grad_c))))[0]
def getRatioC(grad_c, c, dual_var, primal_var, equatn, l_0):
    # line 227: r_c = 1 - sqrt(X/l_0)
    # line 221: X = sum((1-dual(i)) * v(<a, d> + c)) + sum((1+dual(j)) * v(<a, d> + c))
    #           where i in E+(d), I+(d), j in I-(d) 
    X = 0
    m, n = grad_c.shape
    for i in range(m):
        x_new = grad_c[i, :].dot(primal_var) + c[i, 0]
        if x_new > 0:
            X += (1-dual_var[i]) * x_new
        elif x_new < 0 and equatn[i] == True:            
            X += (1+dual_var[i]) * np.abs(x_new)
    return 1-np.sqrt(np.abs(X / (l_0 + 1e-8)))
def getRatio(grad_c, c, grad_f,  primal_var, dual_var, delta, equatn, l_0):
    # Line 199, formula 2.16.
    # When rho is set to 0, it calculates ratio_fea, or it calculates ratio_obj
    primal_obj = getPrimalObject(primal_var, grad_f,  equatn)
    dual_obj = getDualObject(grad_c, grad_f,  c, dual_var, delta)
    
    up = l_0 - primal_obj
    down = (l_0 - dual_obj)
    down += 1e-5

    return up/down






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



    
def boostnlp(f, grad_f, L, x, delta, cuter_problem, step='ls', f_tol=1e-6, time_tol=np.inf, align_tol=1e-3, K=500, traffic=False):
    
    dual_var = np.zeros(cuter_problem.m)
    primal_var = np.zeros(cuter_problem.n)
    l_0 = l0(cuter_problem.cons(x), cuter_problem.is_eq_cons)
    
    if traffic == False:
        values, times, oracles, gaps = [f], [0], [0], [np.dot(grad_f, x-lmo(grad_f))]
        f_improv = np.inf

        x = lmo(grad_f, cuter_problem.n, delta)
        values.append(f)
        oracles.append(1)
    else:
        values, times, oracles, gaps = [f], [0], [0], []
        f_improv = np.inf
    
    while f_improv > f_tol and np.sum(times) < time_tol:
                
        f_old = f
        grad_f_x = grad_f
        gaps.append(np.dot(grad_f_x, x-lmo(grad_f_x)))
        g, num_oracles, align_g = nnmp(x, grad_f_x, align_tol, K, traffic)
        
        if step == 'L':
            gamma = min(align_g*np.linalg.norm(grad_f_x)/(L*np.linalg.norm(g)), 1)
            x = x+gamma*g
        else:
            gamma = min(-np.dot(g, grad_f_x)/np.dot(g, (np.dot(step, g))), 1)
            x = x+gamma*g
        
        values.append(f)
        oracles.append(num_oracles)
        f_improv = f_old-f
        
        
        
        
        
    return gamma, x, l_0



def get_KKT(grad_c, c, grad_f, eta):
        """
        Calcuate KKT error
        :param grad_c: Jacobian of constraints
        :param c: c(x) constraint function value
        :param grad_f: gradient of objective
        :param eta: multiplier in canonical form dual problem
        :return: kkt error
        """

        err_grad = np.max(np.abs(grad_c.T.dot(eta) + grad_f))    
        err_complement = np.max(np.abs(eta * c))
        return max(err_grad, err_complement)



def lmo(gradf, n, TrustRegRad):
    V = TrustRegRad*np.identity(2*n)
    return V[np.argmin(gradf)]


def get_f_gradf_c_gradc_violation(cuter_problem, x):
    """
    Get objective function value, gradient, constraint function value and gradient
    :param cuter_problem: cuter problem instance
    :param x: point to evaluate
    :return: f, grad_f, c, grad_c
    """
    
    f, grad_f = cuter_problem.obj(x, gradient=True)
    c, grad_c = cuter_problem.cons(x, gradient=True)
    violation = v_x(c, cuter_problem.is_eq_cons)
    return f, grad_f, c, grad_c, violation



def linearSolveTrustRegion(cuter_problem, setup_args_dict):
    """
    Non linear solver for cuter problems
    :param cuter instance
    :param setup_args_dict: dust parameter class instance
    :param logger: logger instance to store log information
    :return:
        status:
                -1 - max iteration reached
                1 - solve the problem to optimality
    """         

    x_0 = setup_args_dict['x']
    adjusted_equatn = cuter_problem.is_eq_cons
    zero_d = np.zeros(x_0.shape)
    i, status = 0, -1
    x_k = x_0.copy()
    L = 0
    
    f, grad_f, c, grad_c, violation = get_f_gradf_c_gradc_violation(cuter_problem, x_k)
    max_iter = setup_args_dict['max_iter']
    delta = setup_args_dict['init_delta']; 
    
    
    while i < max_iter:
        phi = get_phi(x_k, cuter_problem, setup_args_dict)
        # Subproblem here.
        l_0_x_k, subg_0_x_k = linearModelPenalty(grad_c, c, grad_f, zero_d, adjusted_equatn)
        
        d_k, x_k, l_0 = boostnlp(phi, subg_0_x_k, L , x_k , delta, cuter_problem, setup_args_dict, step='ls', f_tol=1e-6, time_tol=np.inf, align_tol=1e-3, K=500, traffic=False)
        
        
        l_d_x_k, subg_d_x_k = linearModelPenalty(grad_c, c, grad_f, d_k, adjusted_equatn)
        delta_linearized_model = l_0_x_k - l_d_x_k
        
        kkt_error_k = get_KKT(grad_c, c, grad_f, dual_var)
        
        init_kkt = max(kkt_error_k, 1)
        ratio_opt = getRatio(grad_c, c, grad_f, primal, dual_var, delta, cuter_problem.is_eq_cons, l_0)
        # Update delta.
        if ratio_opt > 0:
            sigma = get_delta_phi(x_k, x_k+d_k, cuter_problem, setup_args_dict) / (delta_linearized_model + 1e-5)
            if np.isnan(sigma):
                # Set it to a very small value to escape inf case.
                sigma = -0x80000000
            if sigma < setup_args_dict['SIGMA']:
                delta = max(0.5*delta, setup_args_dict['MIN_delta'])
            elif sigma > setup_args_dict['DELTA']:
                delta = min(2*delta, setup_args_dict['MAX_delta'])
                
        
        
        # ratio_opt: 3.6. It's actually r_v in paper.
        f, grad_f, c, grad_c, violation = get_f_gradf_c_gradc_violation(cuter_problem, x_k)
        kkt_error_k = get_KKT(grad_c, c, grad_f, dual_var) / init_kkt
        if kkt_error_k < setup_args_dict['eps_opt'] and violation < setup_args_dict['eps_violation']:
            status = 1
            break
        i += 1
