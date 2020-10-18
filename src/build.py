from typing import Dict
import pyomo
import pickle
from pyomo.core.base.PyomoModel import ConcreteModel
import sympy as sp
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

def load_skeleton(skel_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict

def build_model(skel_dict) -> ConcreteModel:
    """
    Builds a pyomo model from a given saved skeleton dictionary
    """
    links = skel_dict["links"]
    positions = skel_dict["positions"]
    dofs = skel_dict["dofs"]
    rot_dict = {}
    pose_dict = {}
    L = len(positions)

    phi     = [sp.symbols(f"\\phi_{{{l}}}")   for l in range(L)]
    theta   = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi     = [sp.symbols(f"\\psi_{{{l}}}")   for l in range(L)]

    for part in dofs:
        rot_dict[part] = sp.eye(3)
        if dofs[part][1]:
            rot_dict[part] = rot_y(theta[0]) @ rot_dict[part]
        if dofs[part][0]:
            rot_dict[part] = rot_x(phi[0]) @ rot_dict[part]
        if dofs[part][2]:
            rot_dict[part] = rot_z(psi[0]) @ rot_dict[part]
        
        rot_dict[part + "_i"] = rot_dict[part].T
    
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")

    for link in links:
        if len(link) == 1:
            pose_dict[link[0]] = sp.Matrix([x, y, z])
        else:
            if link[0] not in pose_dict:
                pose_dict[link[0]] = sp.Matrix([x, y, z])

            translation_vec = sp.Matrix([positions[link[1]][0] - positions[link[0]][0],
                 positions[link[1]][1] - positions[link[0]][1],
                 positions[link[1]][2] - positions[link[0]][2]])
            pose_dict[link[1]] = pose_dict[link[0]] + rot_dict[link[0] + "_i"] @ translation_vec
    
    t_poses = []
    for pose in pose_dict:
        t_poses.append(pose_dict[pose].T)
    
    t_poses_mat = sp.Matrix(t_poses)

    func_map = {"sin":sin, "cos":cos, "ImmutableDenseMatrix":np.array} 
    sym_list = [x, y, z, *phi, *theta, *psi]
    pose_to_3d = sp.lambdify(sym_list, t_poses_mat, modules=[func_map])
    pos_funcs = []

    for i in range(t_poses_mat.shape[0]):
        lamb = sp.lambdify(sym_list, t_poses_mat[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    return(t_poses_mat)

# --- OUTLIER REJECTING COST FUNCTION (REDESCENDING LOSS) ---

def func_step(start, x):
        return 1/(1+np.e**(-1*(x - start)))

def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)
    
def redescending_loss(err, a, b, c):
    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e))/2*e**2
    cost += func_piece(a, b, e)*(a*e - (a**2)/2)
    cost += func_piece(b, c, e)*(a*b - (a**2)/2 + (a*(c-b)/2)*(1-((c-e)/(c-b))**2))
    cost += func_step(c, e)*(a*b - (a**2)/2 + (a*(c-b)/2))
    return cost

# --- Rotation matrices for x, y, and z axes ---

def rot_x(x):
    c = sp.cos(x)
    s = sp.sin(x)
    return sp.Matrix([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def rot_y(y):
    c = sp.cos(y)
    s = sp.sin(y)
    return sp.Matrix([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def rot_z(z):
    c = sp.cos(z)
    s = sp.sin(z)
    return sp.Matrix([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

# --- Numpy equivalent rotation matrices ---

def np_rot_x(x):
    c = np.cos(x)
    s = np.sin(x)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def np_rot_y(y):
    c = np.cos(y)
    s = np.sin(y)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def np_rot_z(z):
    c = np.cos(z)
    s = np.sin(z)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

# --- Reprojection functions ---

def pt3d_to_2d(x, y, z, K, D, R, t):
    x_2d = x*R[0,0] + y*R[0,1] + z*R[0,2] + t.flatten()[0]
    y_2d = x*R[1,0] + y*R[1,1] + z*R[1,2] + t.flatten()[1]
    z_2d = x*R[2,0] + y*R[2,1] + z*R[2,2] + t.flatten()[2]
    #project onto camera plane
    a = x_2d/z_2d
    b = y_2d/z_2d
    #fisheye params
    r = (a**2 + b**2 +1e-12)**0.5 
    th = atan(r)
    #distortion
    th_D = th * (1 + D[0]*th**2 + D[1]*th**4 + D[2]*th**6 + D[3]*th**8)
    x_P = a*th_D/r
    y_P = b*th_D/r
    u = K[0,0]*x_P + K[0,2]
    v = K[1,1]*y_P + K[1,2]
    return u, v

def pt3d_to_x2d(x, y, z, K, D, R, t):
    u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
    return u

def pt3d_to_y2d(x, y, z, K, D, R, t):
    v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
    return v

if __name__ == "__main__":
    skelly = load_skeleton("C://Users//user-pc//Documents//Scripts//FYP//skeletons//plswork.pickle")
    print(skelly)
    print(build_model(skelly))