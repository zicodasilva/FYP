import pyomo
import pickle

def load_skeleton(skel_file) -> dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict

def build_model(skel_dict) -> pyomo:
    """
    Builds a pyomo model from a given saved skeleton dictionary
    """
    #TODO
    return

if __name__ == "__main__":
    print(load_skeleton("C://Users//user-pc//Documents//Scripts//FYP//skeletons//plswork.pickle"))