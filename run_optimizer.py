from simulation_utils import create_env, perform_best
import sys

task = 'Tosser'
w = [0.29754784,0.03725074,0.00664673,0.80602143]
iter_count = 5  # the optimization is nonconvex, so you can specify the number of random starting points

##### YOU DO NOT NEED TO MODIFY THE CODE BELOW THIS LINE #####

D = create_env(task.lower())
perform_best(D, w, iter_count)
