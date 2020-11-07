from pickle import load
from typing import Dict
import pickle
from pyomo.core.base.constraint import ConstraintList
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
import matplotlib.pyplot as plt
import build as bd
import analyse as an

def load_pickle(pickle_file):
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    #print(data["x"][1])
    #print(data["positions"][2])
    return(data)

def compare_plots() -> None:
    scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT//scene_sba.json"
    project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//GT"
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))
    triangulate_func = calib.triangulate_points_fisheye
    #points_2d_filtered_df = points_2d_df[points_2d_df['likelihood']>0.5]
    #print(points_2d_df)
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
    pts = points_3d_df[points_3d_df["frame"]==str(109)][["x", "y", "z", "marker"]].values
    print(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pt in pts:
        ax.scatter(pt[0], pt[1], pt[2], c="b")

    scene_path = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC//scene_sba.json"
    project_dir = "C://Users//user-pc//Documents//Scripts//FYP_tests//DLC"
    df_paths = sorted(glob.glob(os.path.join(project_dir, '*.h5')))
    points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
    D_arr = D_arr.reshape((-1,4))
    triangulate_func = calib.triangulate_points_fisheye
    points_3d_df = calib.get_pairwise_3d_points_from_df(points_2d_df, K_arr, D_arr, R_arr, t_arr, triangulate_func)
    pts = points_3d_df[points_3d_df["frame"]==109][["x", "y", "z", "marker"]].values
    print(pts)
    for pt in pts:
        ax.scatter(pt[0], pt[1], pt[2], c="r")

    pose_dict = {}
    currdir = "C://Users/user-pc/Documents/Scripts/FYP"
    skel_name = "cheetah_serious"
    skelly_dir = os.path.join(currdir, "skeletons", (skel_name + ".pickle"))
    results_dir = os.path.join(currdir, "data", "results", (skel_name + ".pickle"))

    skel_dict = bd.load_skeleton(skelly_dir)
    results = an.load_pickle(results_dir)
    links = skel_dict["links"]
    markers = skel_dict["markers"]
    frame = 29
    for i in range(len(markers)):
        pose_dict[markers[i]] = [results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2]]
        ax.scatter(results["positions"][frame][i][0], results["positions"][frame][i][1], results["positions"][frame][i][2], c="g")
            
    for link in links:
        if len(link)>1:
            ax.plot3D([pose_dict[link[0]][0], pose_dict[link[1]][0]],
            [pose_dict[link[0]][1], pose_dict[link[1]][1]],
            [pose_dict[link[0]][2], pose_dict[link[1]][2]])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()
    print(pts)

def get_error(part1, part2) -> float:
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    data = load_pickle(pickle_path)
    xerrs = []
    yerrs = []
    sum=0
    for frame in range(len(data["x"])):
        sum+= (abs(data["x"][frame,part1,part2]))
        xerrs.append(data["x"][frame,part1,part2])
        yerrs.append(data["y"][frame,part1,part2])
    
    #xstd = np.std(xerrs)
    #ystd = np.std(yerrs)
    #return(xstd, ystd)
    #xrmse = np.sqrt(np.mean(xerrs))
    #yrmse = np.sqrt(np.mean(yerrs))
    return(sum/len(data["x"]))

def hist(part1,part2):
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    data = load_pickle(pickle_path)
    xerrs = []
    yerrs = []
    for frame in range(len(data["x"])):
        #sum+= (data["x"][frame,part1,part2])**2
        xerrs.append(data["x"][frame,part1,part2])
        yerrs.append(data["y"][frame,part1,part2])
    
    plt.hist(xerrs, bins=30)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.show()

if __name__=="__main__":
    pickle_path = "C://Users//user-pc//Desktop//pw_errors.pickle"
    compare_plots()
    """
    data = load_pickle(pickle_path)
    print(get_error(14,15))
    
    print(data["x"][373,11,12])
    print(data["y"][373,11,12])
    print(get_error(12,11))
    for i in range(25):
        for j in range(25):
            print(get_error(i,j))
    """
