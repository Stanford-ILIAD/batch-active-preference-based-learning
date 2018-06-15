import demos
import sys

task   = sys.argv[1].lower()
method = sys.argv[2].lower()
N = int(sys.argv[3])
M = int(sys.argv[4])

if method == 'nonbatch' or method == 'random':
    demos.nonbatch(task, method, N, M)
elif method == 'greedy' or method == 'medoids' or method == 'boundary_medoids' or method == 'successive_elimination':
    b = int(sys.argv[5])
    demos.batch(task, method, N, M, b)
else:
    print('There is no method called ' + method)

