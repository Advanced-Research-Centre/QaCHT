
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Choi
import matplotlib.pyplot as plt

def oracle_type(theta, type):
    qc = QuantumCircuit(2)
    if type == 'cry':
        qc.cry(theta, [0], [1])
    else:
        qc.id([0])
        qc.id([1])
    
    return qc

def DeltaT(dm_i, dm_j):
    diff = dm_i - dm_j
    diff_dag = diff.conjugate().transpose()
    dist = np.real(0.5* np.trace(np.sqrt(np.matmul(diff_dag,diff))))
    return dist

def DeltaB(dm_i, dm_j):
    fid = np.trace(np.sqrt( np.matmul(np.sqrt(dm_i), np.matmul(dm_j,np.sqrt(dm_i))) ))**2
    dist = np.sqrt(2*(1-np.sqrt(fid))) 
    return dist

def DeltaHS(dm_i, dm_j):
    dist = np.real(np.trace((dm_i - dm_j)**2))
    return dist

if __name__ == "__main__":
    qc_id = oracle_type(0, 'id') 
    id_choi_mat = Choi(qc_id).data
    mxd_choi_mat = np.eye(len(id_choi_mat))/len(id_choi_mat)
    theta_range = np.arange(0, 8*np.pi, 0.2)
    d1 = []
    d2 = []
    d3 = []
    for t in theta_range:
        qc_cry = oracle_type(t, 'cry')
        cry_choi_op = Choi(qc_cry).data
        # dist1 = DeltaT(id_choi_mat,cry_choi_op)
        # dist2 = DeltaB(id_choi_mat,cry_choi_op)
        # dist3 = DeltaHS(id_choi_mat,cry_choi_op)
        dist1 = DeltaT(mxd_choi_mat,cry_choi_op)
        dist2 = DeltaB(mxd_choi_mat,cry_choi_op)
        dist3 = DeltaHS(mxd_choi_mat,cry_choi_op)
        d1.append(dist1)
        d2.append(dist2)
        d3.append(dist3)
    
    plt.plot(theta_range, d1, label = "trace", linestyle="-")
    plt.plot(theta_range, d2, label = "bures", linestyle="-")
    plt.plot(theta_range, d3, label = "hilbert-schmidt", linestyle="-")
    # plt.ylabel("Process Distance of Choi CRY(theta) and Choi IxI")
    plt.ylabel("Process Distance of Choi CRY(theta) and Choi mixed")
    plt.xlabel("theta")
    plt.legend()
    plt.show()