import pickle

def load_pickle(pickle_file):
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    print(data)

if __name__=="__main__":
    load_pickle("C://Users//user-pc//Documents//Scripts//FYP//data//results//traj_results.pickle")
