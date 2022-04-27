import numpy as np
from qiskit import *
import scipy.linalg as la
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.operators import Chi, Choi
from qiskit.quantum_info.operators import Operator, process_fidelity
from investigation import *
pi = np.pi

def oracle_type(theta, type):
    qc = QuantumCircuit(2)
    if type == 'cry':
        qc.cry(theta, [0], [1])
    else:
        qc.id([0])
        qc.id([1])
    
    return qc

if __name__ == "__main__":
    qc_id = oracle_type(0, 'id') 
    id_choi_mat = Choi(qc_id).data
    id_chi_op = Operator(Chi(qc_id).data)
    theta_range = np.arange(0, 4*pi, 0.2)
    x = []
    for t in theta_range:
        qc_cry = oracle_type(t, 'cry')
        cry_chi_op = Operator(Chi(qc_cry).data)
        process_distance = 1 - process_fidelity( id_chi_op, cry_chi_op )
        x.append(process_distance)
    
    plt.plot(theta_range, x)
    plt.show()