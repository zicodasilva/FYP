from typing import Dict, Tuple
import pickle
from pyomo.core.base.constraint import Constraint, ConstraintList
import sympy as sp
import numpy as np
import os
import glob
from calib import utils, calib, plotting, app, extract
from scipy import stats
from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.PyomoModel import ConcreteModel
# Bring your packages onto the path
import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
from AcinoSet.src.lib import utils as utility, misc, vid

def generate_plotting_data(results_dir: str, scene_file: str, m: ConcreteModel, poses: Dict) -> None:
    # GET ALL THE PLOTTING DATA AND SAVE
    x_opt = []
    for n in m.N:
        x_opt.append([value(m.x[n, p]) for p in m.P])
    x_opt = np.array(x_opt)

    scatter_frames = []
    lines_frames = []
    for idx, states in enumerate(x_opt):
        skeleton = poses(*states)
        scatter_frames.append(skeleton)
        lines_frames.append(skeleton[[0,1,0,2,1,2,1,3,0,3,2,3,3,4,4,5,5,6,6,7,3,8,4,8,8,9,9,10,3,11,4,11,11,12,12,13,4,14,5,14,14,15,15,16,4,17,5,17,17,18,18,19], :])

    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_file)
    D_arr = D_arr.reshape((-1,4))
    np.save(os.path.join(results_dir, 'plot_data.npy'), [scatter_frames,lines_frames,K_arr,D_arr,R_arr,t_arr])
    print(f"saved plot_data.npy\n")

def load_data(file) -> Dict:
    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return data

def build_model(project_dir, dlc_thresh=0.5) -> Tuple[ConcreteModel, Dict]:
    #SYMBOLIC ROTATION MATRIX FUNCTIONS
    print("Generate and load data...")

    L = 14  #number of joints in the cheetah model

    # defines arrays ofa angles, velocities and accelerations
    phi     = [sp.symbols(f"\\phi_{{{l}}}")   for l in range(L)]
    theta   = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi     = [sp.symbols(f"\\psi_{{{l}}}")   for l in range(L)]

    #ROTATIONS
    # head
    RI_0 = rot_z(psi[0]) @ rot_x(phi[0]) @ rot_y(theta[0])
    R0_I = RI_0.T
    # neck
    RI_1 = rot_z(psi[1]) @ rot_x(phi[1]) @ rot_y(theta[1]) @ RI_0
    R1_I = RI_1.T
    # front torso
    RI_2 = rot_y(theta[2]) @ RI_1
    R2_I = RI_2.T
    # back torso
    RI_3 = rot_z(psi[3])@ rot_x(phi[3]) @ rot_y(theta[3]) @ RI_2
    R3_I = RI_3.T
    # tail base
    RI_4 = rot_z(psi[4]) @ rot_y(theta[4]) @ RI_3
    R4_I = RI_4.T
    # tail mid
    RI_5 = rot_z(psi[5]) @ rot_y(theta[5]) @ RI_4
    R5_I = RI_5.T
    #l_shoulder
    RI_6 = rot_y(theta[6]) @ RI_2
    R6_I = RI_6.T
    #l_front_knee
    RI_7 = rot_y(theta[7]) @ RI_6
    R7_I = RI_7.T
    #r_shoulder
    RI_8 = rot_y(theta[8]) @ RI_2
    R8_I = RI_8.T
    #r_front_knee
    RI_9 = rot_y(theta[9]) @ RI_8
    R9_I = RI_9.T
    #l_hip
    RI_10 = rot_y(theta[10]) @ RI_3
    R10_I = RI_10.T
    #l_back_knee
    RI_11 = rot_y(theta[11]) @ RI_10
    R11_I = RI_11.T
    #r_hip
    RI_12 = rot_y(theta[12]) @ RI_3
    R12_I = RI_12.T
    #r_back_knee
    RI_13 = rot_y(theta[13]) @ RI_12
    R13_I = RI_13.T

    # defines the position, velocities and accelerations in the inertial frame
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")


    # SYMBOLIC CHEETAH POSE POSITIONS
    p_head          = sp.Matrix([x, y, z])

    p_l_eye         = p_head         + R0_I  @ sp.Matrix([0, 0.03, 0])
    p_r_eye         = p_head         + R0_I  @ sp.Matrix([0, -0.03, 0])
    p_nose          = p_head         + R0_I  @ sp.Matrix([0.055, 0, -0.055])

    p_neck_base     = p_head         + R1_I  @ sp.Matrix([-0.28, 0, 0])
    p_spine         = p_neck_base    + R2_I  @ sp.Matrix([-0.37, 0, 0])

    p_tail_base     = p_spine        + R3_I  @ sp.Matrix([-0.37, 0, 0])
    p_tail_mid      = p_tail_base    + R4_I  @ sp.Matrix([-0.28, 0, 0])
    p_tail_tip      = p_tail_mid     + R5_I  @ sp.Matrix([-0.36, 0, 0])

    p_l_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, 0.08, -0.10])
    p_l_front_knee  = p_l_shoulder   + R6_I  @ sp.Matrix([0, 0, -0.24])
    p_l_front_ankle = p_l_front_knee + R7_I  @ sp.Matrix([0, 0, -0.28])

    p_r_shoulder    = p_neck_base    + R2_I  @ sp.Matrix([-0.04, -0.08, -0.10])
    p_r_front_knee  = p_r_shoulder   + R8_I  @ sp.Matrix([0, 0, -0.24])
    p_r_front_ankle = p_r_front_knee + R9_I  @ sp.Matrix([0, 0, -0.28])

    p_l_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, 0.08, -0.06])
    p_l_back_knee   = p_l_hip        + R10_I @ sp.Matrix([0, 0, -0.32])
    p_l_back_ankle  = p_l_back_knee  + R11_I @ sp.Matrix([0, 0, -0.25])

    p_r_hip         = p_tail_base    + R3_I  @ sp.Matrix([0.12, -0.08, -0.06])
    p_r_back_knee   = p_r_hip        + R12_I @ sp.Matrix([0, 0, -0.32])
    p_r_back_ankle  = p_r_back_knee  + R13_I @ sp.Matrix([0, 0, -0.25])

    positions = sp.Matrix([
        p_l_eye.T, p_r_eye.T, p_nose.T,
        p_neck_base.T, p_spine.T,
        p_tail_base.T, p_tail_mid.T, p_tail_tip.T,
        p_l_shoulder.T, p_l_front_knee.T, p_l_front_ankle.T,
        p_r_shoulder.T, p_r_front_knee.T, p_r_front_ankle.T,
        p_l_hip.T, p_l_back_knee.T, p_l_back_ankle.T,
        p_r_hip.T, p_r_back_knee.T, p_r_back_ankle.T
    ])

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    func_map = {"sin":sin, "cos":cos, "ImmutableDenseMatrix":np.array}
    sym_list = [x, y, z, *phi, *theta, *psi]
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []
    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i,:], modules=[func_map])
        pos_funcs.append(lamb)

    scene_path = os.path.join(project_dir, "scene_sba.json")

    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))

    # markers = misc.get_markers()
    markers = dict(enumerate([
    "l_eye", "r_eye", "nose",
    "neck_base", "spine",
    "tail_base", "tail1", "tail2",
    "l_shoulder", "l_front_knee", "l_front_ankle",
    "r_shoulder", "r_front_knee", "r_front_ankle",
    "l_hip", "l_back_knee", "l_back_ankle",
    "r_hip", "r_back_knee", "r_back_ankle"
    ]))

    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        d_idx = {1:"x", 2:"y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"]== n-1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"]== c-1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]

    # Parameters

    h = 1/120 #timestep
    end_frame = 165
    start_frame = 50
    N = end_frame-start_frame # N > start_frame !!
    P = 3 + len(phi)+len(theta)+len(psi)
    L = len(pos_funcs)
    C = len(K_arr)
    D2 = 2
    D3 = 3
    W = 2

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    # measurement standard deviation
    R = np.array([
        1.24,
        1.18,
        1.2,
        2.08,
        2.04,
        2.52,
        2.73,
        1.83,
        3.4,
        2.91,
        2.85,
        # 2.27, # l_front_paw
        3.47,
        2.75,
        2.69,
        # 2.24, # r_front_paw
        3.53,
        2.69,
        2.49,
        # 2.34, # l_back_paw
        3.26,
        2.76,
        2.33,
        # 2.4, # r_back_paw
    ])
    R_pw = np.repeat(7, len(R))
    Q = np.array([ # model parameters variance
        4.0,
        7.0,
        5.0,
        13.0,
        32.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        9.0,
        18.0,
        43.0,
        53.0,
        90.0,
        118.0,
        247.0,
        186.0,
        194.0,
        164.0,
        295.0,
        243.0,
        334.0,
        149.0,
        26.0,
        12.0,
        0.0,
        34.0,
        43.0,
        51.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])**2

    triangulate_func = calib.triangulate_points_fisheye
    points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>dlc_thresh]
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_filtered_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)

    # estimate initial points
    nose_pts = points_3d_df[points_3d_df["marker"]=="nose"][["x", "y", "z", "frame"]].values
    x_slope, x_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,0])
    y_slope, y_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,1])
    z_slope, z_intercept, *_ = stats.linregress(nose_pts[:,3], nose_pts[:,2])
    frame_est = np.arange(N)
    x_est = frame_est*x_slope + x_intercept
    y_est = frame_est*y_slope + y_intercept
    z_est = frame_est*z_slope + z_intercept
    print(x_est.shape)
    psi_est = np.arctan2(y_slope, x_slope)

    print("Build optimisation problem - Start")
    m = ConcreteModel(name = "Skeleton")

    # ===== SETS =====
    m.N = RangeSet(N) #number of timesteps in trajectory
    m.P = RangeSet(P) #number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n)
    m.L = RangeSet(L) #number of labels
    m.C = RangeSet(C) #number of cameras
    m.D2 = RangeSet(D2) #dimensionality of measurements
    m.D3 = RangeSet(D3) #dimensionality of measurements
    m.W = RangeSet(W) # Number of pairwise terms to include.

    def init_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n+start_frame, c, l)
        if likelihood > dlc_thresh:
            return 1/R[l-1]
        else:
            return 0
    m.meas_err_weight = Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True)  # IndexError: index 0 is out of bounds for axis 0 with size 0

    def init_pw_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n+start_frame, c, l)
        if likelihood <= dlc_thresh:
            return 1/R_pw[l-1]
        else:
            return 0
    m.meas_pw_err_weight = Param(m.N, m.C, m.L, initialize=init_pw_meas_weights, mutable=True)  # IndexError: index 0 is out of bounds for axis 0 with size 0


    def init_model_weights(m, p):
        if Q[p-1] != 0.0:
            return 1/Q[p-1]
        else:
            return 0
    m.model_err_weight = Param(m.P, initialize=init_model_weights)

    m.h = h

    def init_measurements_df(m, n, c, l, d2):
        return get_meas_from_df(n+start_frame, c, l, d2)
    m.meas = Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df)

    # resultsfilename='C://Users//user-pc//Desktop//pwpoints.pickle'
    # with open(resultsfilename, 'rb') as f:
    #         data=pickle.load(f)
    pw_data = {}
    for cam in range(C):
        pw_data[cam] = load_data(f"/Users/zico/msc/dev/FYP/data/09_03_2019/lily/run/cam{cam+1}-predictions.pickle")

    index_dict = {"nose":23, "r_eye":0, "l_eye":1, "neck_base":24, "spine":6, "tail_base":22, "tail1":11,
     "tail2":12, "l_shoulder":13,"l_front_knee":14,"l_front_ankle":15,"r_shoulder":2,
      "r_front_knee":3, "r_front_ankle":4,"l_hip":17,"l_back_knee":18, "l_back_ankle":19,
       "r_hip":7,"r_back_knee":8,"r_back_ankle":9}

    pair_dict = {"r_eye":[23, 24], "l_eye":[23, 24], "nose":[6, 24], "neck_base":[6, 23], "spine":[22, 24], "tail_base":[6, 11], "tail1":[6, 22],
     "tail2":[11, 22], "l_shoulder":[6, 24],"l_front_knee":[6, 24],"l_front_ankle":[6, 24],"r_shoulder":[6, 24],
      "r_front_knee":[6, 24], "r_front_ankle":[6, 24],"l_hip":[6, 22],"l_back_knee":[6, 22], "l_back_ankle":[6, 22],
       "r_hip":[6, 22],"r_back_knee":[6, 22],"r_back_ankle":[6, 22]}


    def init_pw_measurements(m, n, c, l, d2, w):
        pw_values = pw_data[c-1][n+start_frame]
        marker = markers[l-1]
        base = pair_dict[marker][w-1]
        val = pw_values['pose'][d2-1::3]
        val_pw = pw_values['pws'][:,:,:,d2-1]

        return val[base]+val_pw[0,base,index_dict[marker]]
    m.pw_meas = Param(m.N, m.C, m.L, m.D2, m.W, initialize=init_pw_measurements, within=Any)
    """
    def init_pw_measurements2(m, n, c, l, d2):
        val=0
        if n-1 >= 20 and n-1 < 30:
            fn = 10*(c-1)+(n-20)-1
            x=data[fn]['pose'][0::3]
            y=data[fn]['pose'][1::3]
            xpw=data[fn]['pws'][:,:,:,0]
            ypw=data[fn]['pws'][:,:,:,1]
            marker = markers[l-1]
            if "ankle" in marker:
                base = pair_dict[marker][1]
                if d2==1:
                    val=x[base]+xpw[0,base,index_dict[marker]]
                elif d2==2:
                    val=y[base]+ypw[0,base,index_dict[marker]]
                #sum/=len(pair_dict[marker])
                return val
        else:
            return(0.0)

    m.pw_meas2 = Param(m.N, m.C, m.L, m.D2, initialize=init_pw_measurements2, within=Any)
    """
    # ===== VARIABLES =====
    m.x = Var(m.N, m.P) #position
    m.dx = Var(m.N, m.P) #velocity
    m.ddx = Var(m.N, m.P) #acceleration
    m.poses = Var(m.N, m.L, m.D3)
    m.slack_model = Var(m.N, m.P)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0)
    m.slack_pw_meas = Var(m.N, m.C, m.L, m.D2, m.W, initialize=0.0)


    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N-start_frame, P))
    init_x[:,0] = x_est[start_frame: start_frame+N] #x
    init_x[:,1] = y_est[start_frame: start_frame+N] #y
    init_x[:,2] = z_est[start_frame: start_frame+N] #z
    init_x[:,31] = psi_est #yaw - psi
    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))
    for n in m.N:
        for p in m.P:
            if n<len(init_x): #init using known values
                m.x[n,p].value = init_x[n-1,p-1]
                m.dx[n,p].value = init_dx[n-1,p-1]
                m.ddx[n,p].value = init_ddx[n-1,p-1]
            else: #init using last known value
                m.x[n,p].value = init_x[-1,p-1]
                m.dx[n,p].value = init_dx[-1,p-1]
                m.ddx[n,p].value = init_ddx[-1,p-1]
        #init pose
        var_list = [m.x[n,p].value for p in range(1, P+1)]
        for l in m.L:
            [pos] = pos_funcs[l-1](*var_list)
            for d3 in m.D3:
                m.poses[n,l,d3].value = pos[d3-1]

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m,n,l,d3):
        #get 3d points
        var_list = [m.x[n,p] for p in range(1, P+1)]
        [pos] = pos_funcs[l-1](*var_list)
        return pos[d3-1] == m.poses[n,l,d3]

    m.pose_constraint = Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    def backwards_euler_pos(m,n,p): # position
        if n > 1:
    #             return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n-1,p] + m.h**2 * m.ddx[n-1,p]/2
            return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n,p]

        else:
            return Constraint.Skip
    m.integrate_p = Constraint(m.N, m.P, rule = backwards_euler_pos)


    def backwards_euler_vel(m,n,p): # velocity
        if n > 1:
            return m.dx[n,p] == m.dx[n-1,p] + m.h*m.ddx[n,p]
        else:
            return Constraint.Skip
    m.integrate_v = Constraint(m.N, m.P, rule = backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p):
        if n > 1:
            return m.ddx[n,p] == m.ddx[n-1,p] + m.slack_model[n,p]
        else:
            return Constraint.Skip
    m.constant_acc = Constraint(m.N, m.P, rule = constant_acc)

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2):
        #project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        x, y, z = m.poses[n,l,1], m.poses[n,l,2], m.poses[n,l,3]
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] ==0
    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule = measurement_constraints)

    def pw_measurement_constraints(m, n, c, l, d2, w):
        #project
        K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
        x, y, z = m.poses[n,l,1], m.poses[n,l,2], m.poses[n,l,3]
        return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.pw_meas[n, c, l, d2, w] - m.slack_pw_meas[n, c, l, d2, w] ==0.0
    m.pw_measurement = Constraint(m.N, m.C, m.L, m.D2, m.W, rule = pw_measurement_constraints)
    """
    def pw_measurement_constraints2(m, n, c, l, d2):
        #project
        if n-1 >= 20 and n-1 < 30 and "ankle" in markers[l-1]:
            K, D, R, t = K_arr[c-1], D_arr[c-1], R_arr[c-1], t_arr[c-1]
            x, y, z = m.poses[n,l,1], m.poses[n,l,2], m.poses[n,l,3]
            return proj_funcs[d2-1](x, y, z, K, D, R, t) - m.pw_meas2[n, c, l, d2] - m.slack_meas[n, c, l, d2] ==0.0
        else:
            return(Constraint.Skip)
    m.pw_measurement2 = Constraint(m.N, m.C, m.L, m.D2, rule = pw_measurement_constraints2)
    """

    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    #Head
    def head_psi_0(m,n):
        return abs(m.x[n,4]) <= np.pi/6
    m.head_psi_0 = Constraint(m.N, rule=head_psi_0)
    def head_theta_0(m,n):
        return abs(m.x[n,18]) <= np.pi/6
    m.head_theta_0 = Constraint(m.N, rule=head_theta_0)

    #Neck
    def neck_phi_1(m,n):
        return abs(m.x[n,5]) <= np.pi/6
    m.neck_phi_1 = Constraint(m.N, rule=neck_phi_1)
    def neck_theta_1(m,n):
        return abs(m.x[n,19]) <= np.pi/6
    m.neck_theta_1 = Constraint(m.N, rule=neck_theta_1)
    def neck_psi_1(m,n):
        return abs(m.x[n,33]) <= np.pi/6
    m.neck_psi_1 = Constraint(m.N, rule=neck_psi_1)

    #Front torso
    def front_torso_theta_2(m,n):
        return abs(m.x[n,20]) <= np.pi/6
    m.front_torso_theta_2 = Constraint(m.N, rule=front_torso_theta_2)

    #Back torso
    def back_torso_theta_3(m,n):
        return abs(m.x[n,21]) <= np.pi/6
    m.back_torso_theta_3 = Constraint(m.N, rule=back_torso_theta_3)
    # --- Back torso phi constraint - uncomment if needed! ---
    def back_torso_phi_3(m,n):
        return abs(m.x[n,7]) <= np.pi/6
    m.back_torso_phi_3 = Constraint(m.N, rule=back_torso_phi_3)
    def back_torso_psi_3(m,n):
        return abs(m.x[n,35]) <= np.pi/6
    m.back_torso_psi_3 = Constraint(m.N, rule=back_torso_psi_3)

    #Tail base
    def tail_base_theta_4(m,n):
        return abs(m.x[n,22]) <= np.pi/1.5
    m.tail_base_theta_4 = Constraint(m.N, rule=tail_base_theta_4)
    def tail_base_psi_4(m,n):
        return abs(m.x[n,36]) <= np.pi/1.5
    m.tail_base_psi_4 = Constraint(m.N, rule=tail_base_psi_4)

    #Tail base
    def tail_mid_theta_5(m,n):
        return abs(m.x[n,23]) <= np.pi/1.5
    m.tail_mid_theta_5 = Constraint(m.N, rule=tail_mid_theta_5)
    def tail_mid_psi_5(m,n):
        return abs(m.x[n,37]) <= np.pi/1.5
    m.tail_mid_psi_5 = Constraint(m.N, rule=tail_mid_psi_5)

    #Front left leg
    def l_shoulder_theta_6(m,n):
        return abs(m.x[n,24]) <= np.pi/2
    m.l_shoulder_theta_6 = Constraint(m.N, rule=l_shoulder_theta_6)
    def l_front_knee_theta_7(m,n):
        return abs(m.x[n,25] + np.pi/2) <= np.pi/2
    m.l_front_knee_theta_7 = Constraint(m.N, rule=l_front_knee_theta_7)

    #Front right leg
    def r_shoulder_theta_8(m,n):
        return abs(m.x[n,26]) <= np.pi/2
    m.r_shoulder_theta_8 = Constraint(m.N, rule=r_shoulder_theta_8)
    def r_front_knee_theta_9(m,n):
        return abs(m.x[n,27] + np.pi/2) <= np.pi/2
    m.r_front_knee_theta_9 = Constraint(m.N, rule=r_front_knee_theta_9)

    #Back left leg
    def l_hip_theta_10(m,n):
        return abs(m.x[n,28]) <= np.pi/2
    m.l_hip_theta_10 = Constraint(m.N, rule=l_hip_theta_10)
    def l_back_knee_theta_11(m,n):
        return abs(m.x[n,29] - np.pi/2) <= np.pi/2
    m.l_back_knee_theta_11 = Constraint(m.N, rule=l_back_knee_theta_11)

    #Back right leg
    def r_hip_theta_12(m,n):
        return abs(m.x[n,30]) <= np.pi/2
    m.r_hip_theta_12 = Constraint(m.N, rule=r_hip_theta_12)
    def r_back_knee_theta_13(m,n):
        return abs(m.x[n,31] - np.pi/2) <= np.pi/2
    m.r_back_knee_theta_13 = Constraint(m.N, rule=r_back_knee_theta_13)

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        slack_pw_meas_err = 0.0

        for n in m.N:
            #Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * m.slack_model[n, p] ** 2
            #Measurement Error
            for l in m.L:
                for c in m.C:
                    for d2 in m.D2:
                        slack_meas_err += misc.redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2], 3, 7, 20)
                        for w in m.W:
                            slack_meas_err += misc.redescending_loss(m.meas_pw_err_weight[n, c, l] * m.slack_pw_meas[n, c, l, d2, w], 3, 7, 20)
        return slack_meas_err + slack_model_err

    m.obj = Objective(rule = obj)

    print("Build optimisation problem - End")

    return m, pose_to_3d

def solve_optimisation(model, exe_path = None) -> None:
    """
    Solves a given trajectory optimisation problem given a model and solver
    """
    opt = SolverFactory(
        'ipopt',
        executable=exe_path
    )

    # solver options
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 500
    opt.options["max_cpu_time"] = 3600
    opt.options["tol"] = 1e-1
    opt.options["OF_print_timing_statistics"] = "yes"
    opt.options["OF_print_frequency_iter"] = 10
    opt.options["OF_hessian_approximation"] = "limited-memory"
    opt.options["linear_solver"] = "ma86"

    LOG_DIR = '/Users/zico/msc/dev/FYP/logs'

    # --- This step may take a while! ---
    print("Optimisation - Start")
    opt.solve(
        model, tee=True,
        keepfiles=True,
        logfile=os.path.join(LOG_DIR, "solver.log")
    )
    print("Optimisation - End")

def save_results(model, poses, project_dir: str, dlc_thresh=0.5) -> None:
    result_dir = os.path.join(project_dir, "results")
    scene_path = os.path.join(project_dir, "scene_sba.json")
    save_data(model, file_path=os.path.join(result_dir, 'traj_results.pickle'), poses=poses)
    generate_plotting_data(result_dir, scene_path, model, poses)
    x, dx, ddx =  [], [], []
    for n in model.N:
        x.append([value(model.x[n, p]) for p in model.P])
        dx.append([value(model.dx[n, p]) for p in model.P])
        ddx.append([value(model.ddx[n, p]) for p in model.P])
    save_fte(dict(x=x, dx=dx, ddx=ddx), result_dir, scene_path, 50, dlc_thresh)

def convert_to_dict(m, poses) -> Dict:
    x_optimised = []
    dx_optimised = []
    ddx_optimised = []
    for n in m.N:
        x_optimised.append([value(m.x[n, p]) for p in m.P])
        dx_optimised.append([value(m.dx[n, p]) for p in m.P])
        ddx_optimised.append([value(m.ddx[n, p]) for p in m.P])
    x_optimised = np.array(x_optimised)
    dx_optimised = np.array(dx_optimised)
    ddx_optimised = np.array(ddx_optimised)

    print(poses)
    print(x_optimised)

    positions = np.array([poses(*states) for states in x_optimised])
    file_data = dict(
        positions=positions,
        x=x_optimised,
        dx=dx_optimised,
        ddx=ddx_optimised,
    )
    return file_data

def save_data(file_data, file_path, poses, dict=True) -> None:

    if dict:
        file_data = convert_to_dict(file_data, poses)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(file_data, f)

    print(f'save {file_path}')

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


def save_fte(states, out_dir, scene_fpath, start_frame, dlc_thresh, save_videos=True):
    positions = [misc.get_3d_marker_coords(state) for state in states['x']]
    out_fpath = os.path.join(out_dir, f"fte.pickle")
    markers = dict(enumerate([
    "l_eye", "r_eye", "nose",
    "neck_base", "spine",
    "tail_base", "tail1", "tail2",
    "l_shoulder", "l_front_knee", "l_front_ankle",
    "r_shoulder", "r_front_knee", "r_front_ankle",
    "l_hip", "l_back_knee", "l_back_ankle",
    "r_hip", "r_back_knee", "r_back_ankle"
    ]))

    utility.save_optimised_cheetah(positions, out_fpath, extra_data=dict(**states, start_frame=start_frame))
    utility.save_3d_cheetah_as_2d(positions, out_dir, scene_fpath, markers, calib.project_points_fisheye, start_frame)

    if save_videos:
        video_fpaths = sorted(glob.glob(os.path.join(os.path.dirname(out_dir), 'cam[1-9].mp4'))) # original vids should be in the parent dir
        create_labeled_videos(video_fpaths, out_dir=out_dir, draw_skeleton=True, pcutoff=dlc_thresh)

def create_labeled_videos(video_fpaths, videotype="mp4", codec="mp4v", outputframerate=None, out_dir=None, draw_skeleton=False, pcutoff=0.5, dotsize=6, colormap='jet', skeleton_color='white'):
    from functools import partial
    from multiprocessing import Pool

    print('Saving labeled videos...')

    # bodyparts = misc.get_markers()
    bodyparts = dict(enumerate([
    "l_eye", "r_eye", "nose",
    "neck_base", "spine",
    "tail_base", "tail1", "tail2",
    "l_shoulder", "l_front_knee", "l_front_ankle",
    "r_shoulder", "r_front_knee", "r_front_ankle",
    "l_hip", "l_back_knee", "l_back_ankle",
    "r_hip", "r_back_knee", "r_back_ankle"
    ]))
    bodyparts2connect = misc.get_skeleton() if draw_skeleton else None

    if not video_fpaths:
        print("No videos were found. Please check your paths\n")
        return

    if out_dir is None:
        out_dir = os.path.relpath(os.path.dirname(video_fpaths[0]), os.getcwd())

    func = partial(vid.proc_video, out_dir, bodyparts, codec, bodyparts2connect, outputframerate, draw_skeleton, pcutoff, dotsize, colormap, skeleton_color)

    with Pool(min(os.cpu_count(), len(video_fpaths))) as pool:
        pool.map(func,video_fpaths)

    print('Done!\n')

if __name__ == "__main__":
    project_dir = "/Users/zico/msc/dev/FYP/data/09_03_2019/lily/run"
    model, pose3d = build_model(project_dir)
    solve_optimisation(model)
    save_results(model, pose3d, project_dir)
