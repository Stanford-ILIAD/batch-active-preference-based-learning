import numpy as np
import scipy.optimize as opt
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids
from scipy.spatial import ConvexHull

def func_psi(psi_set, w_samples):
    y = psi_set.dot(w_samples.T)
    term1 = np.sum(1.-np.exp(-np.maximum(y,0)),axis=1)
    term2 = np.sum(1.-np.exp(-np.maximum(-y,0)),axis=1)
    f = -np.minimum(term1,term2)
    return f

def generate_psi(simulation_object, inputs_set):
    z = simulation_object.feed_size
    inputs_set = np.array(inputs_set)
    if len(inputs_set.shape) == 1:
        inputs1 = inputs_set[0:z].reshape(1,z)
        inputs2 = inputs_set[z:2*z].reshape(1,z)
        input_count = 1
    else:
        inputs1 = inputs_set[:,0:z]
        inputs2 = inputs_set[:,z:2*z]
        input_count = inputs_set.shape[0]
    d = simulation_object.num_of_features
    features1 = np.zeros([input_count, d])
    features2 = np.zeros([input_count, d])  
    for i in range(input_count):
        simulation_object.feed(list(inputs1[i]))
        features1[i] = simulation_object.get_features()
        simulation_object.feed(list(inputs2[i]))
        features2[i] = simulation_object.get_features()
    psi_set = features1 - features2
    return psi_set

def func(inputs_set, *args):
    simulation_object = args[0]
    w_samples = args[1]
    psi_set = generate_psi(simulation_object, inputs_set)
    return func_psi(psi_set, w_samples)

def nonbatch(simulation_object, w_samples):
    z = simulation_object.feed_size
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    opt_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*z)), args=(simulation_object, w_samples), bounds=simulation_object.feed_bounds*2, approx_grad=True)
    return opt_res[0][0:z], opt_res[0][z:2*z]


def select_top_candidates(simulation_object, w_samples, B):
    d = simulation_object.num_of_features
    z = simulation_object.feed_size

    inputs_set = np.zeros(shape=(0,2*z))
    psi_set = np.zeros(shape=(0,d))
    f_values = np.zeros(shape=(0))
    data = np.load('ctrl_samples/' + simulation_object.name + '.npz')
    inputs_set = data['inputs_set']
    psi_set = data['psi_set']
    f_values = func_psi(psi_set, w_samples)
    id_input = np.argsort(f_values)
    inputs_set = inputs_set[id_input[0:B]]
    psi_set = psi_set[id_input[0:B]]
    f_values = f_values[id_input[0:B]]
    return inputs_set, psi_set, f_values, d, z

def greedy(simulation_object, w_samples, b):
    inputs_set, _, _, _, z = select_top_candidates(simulation_object, w_samples, b)
    return inputs_set[:, :z], inputs_set[:, z:]

def medoids(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, _, _, z = select_top_candidates(simulation_object, w_samples, B)

    D = pairwise_distances(psi_set, metric='euclidean')
    M, C = kmedoids.kMedoids(D, b)
    return inputs_set[M, :z], inputs_set[M, z:]

def boundary_medoids(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, _, _, z = select_top_candidates(simulation_object, w_samples, B)

    hull = ConvexHull(psi_set)
    simplices = np.unique(hull.simplices)
    boundary_psi = psi_set[simplices]
    boundary_inputs = inputs_set[simplices]
    D = pairwise_distances(boundary_psi, metric='euclidean')
    M, C = kmedoids.kMedoids(D, b)
    
    return boundary_inputs[M, :z], boundary_inputs[M, z:]

def successive_elimination(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, f_values, d, z = select_top_candidates(simulation_object, w_samples, B)

    D = pairwise_distances(psi_set, metric='euclidean')
    D = np.array([np.inf if x==0 else x for x in D.reshape(B*B,1)]).reshape(B,B)
    while len(inputs_set) > b:
        ij_min = np.where(D == np.min(D))
        if len(ij_min) > 1 and len(ij_min[0]) > 1:
            ij_min = ij_min[0]
        elif len(ij_min) > 1:
            ij_min = np.array([ij_min[0],ij_min[1]])

        if f_values[ij_min[0]] < f_values[ij_min[1]]:
            delete_id = ij_min[1]
        else:
            delete_id = ij_min[0]
        D = np.delete(D, delete_id, axis=0)
        D = np.delete(D, delete_id, axis=1)
        f_values = np.delete(f_values, delete_id)
        inputs_set = np.delete(inputs_set, delete_id, axis=0)
        psi_set = np.delete(psi_set, delete_id, axis=0)
    return inputs_set[:,0:z], inputs_set[:,z:2*z]

def random(simulation_object, w_samples):
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    input_A = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    input_B = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    return input_A, input_B
