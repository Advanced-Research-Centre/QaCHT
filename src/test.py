from qiskit import *
import numpy
from qiskit.quantum_info import partial_trace
theta = numpy.pi/2

register_size = 4
swap = QuantumCircuit(2)
def oracle(theta):
    swap.rxx(theta, 0, 1)
    swap.ryy(theta, 0, 1)
    return swap.to_gate(label = f'iSWAP-{theta}').control(1)

def identity():
    qc1 =  QuantumCircuit(11) 
    qc1.x(0)
    qc1.x(1)
    qc1.x(2)
    qc1.x(3)
    qc1.append(oracle(0), [10,0,4])
    qc1.append(oracle(0), [10,1,5])
    qc1.append(oracle(0), [10,2,6])
    qc1.append(oracle(0), [10,3,7])
    return qc1

def alternate(theta):
    qc2  =  QuantumCircuit(11)
    # qc2.h(0)
    # qc2.h(1)
    # qc2.h(2)
    # qc2.h(3)
    qc2.x(10)
    qc2.append(oracle(theta), [10,0,4])
    qc2.append(oracle(theta), [10,1,5])
    qc2.append(oracle(theta), [10,2,6])
    qc2.append(oracle(theta), [10,3,7])
    return qc2



dict_prob_qc1 = {}
dict_prob_qc2 = {}
# qc = identity()
# print(qc)
simulator = Aer.get_backend('statevector_simulator')
# result_qc1 = execute(qc, simulator).result()
# statevector_qc1 = result_qc1.get_statevector( qc ).data
# statevector_qc1 = partial_trace(statevector_qc1, [8, 9, 10])
# statevector_qc1 = numpy.diag(statevector_qc1.data)
# for i in range(len(statevector_qc1)):
#     p = abs(statevector_qc1[i])
#     if p > 1e-5:
#         dict_prob_qc1[f'{bin(i)[2:].zfill( 4 )}'] = p**2


for theta in [0]:#numpy.arange(0, 2*numpy.pi, numpy.pi/10):
    qc = alternate(theta)
    result_qc2 = execute( qc, simulator).result()
    # print(qc)
    # exit()
    statevector_qc2 = result_qc2.get_statevector(qc ).data
    statevector_qc2 = partial_trace(statevector_qc2, [8, 9, 10])
    statevector_qc2 = numpy.diag(statevector_qc2.data)

    for i in range(len(statevector_qc2)):
        p = abs(statevector_qc2[i])
        if p > 1e-5:
            dict_prob_qc2[f'{bin(i)[2:].zfill( 8 )}'] = p**2

    err_prob = 0
    for k in dict_prob_qc2.keys():
        
        if k[:register_size] == k[register_size:]:
            err_prob+= dict_prob_qc2[k]
        # else:
            # print('BAPI BARI JA')
    print(theta, err_prob)

    # list_key_hypo_1 = []
    # list_key_hypo_2 = []
    # for k in dict_prob_qc1.keys():
    #     list_key_hypo_1.append(k)
    # for k in dict_prob_qc2.keys():
    #     list_key_hypo_2.append(k)


    # common_bits = []
    # for i in list_key_hypo_1:
    #     for j in list_key_hypo_2:
    #         if i == j:
    #             common_bits.append(i)
    # x = 0
    # for k in list_key_hypo_2:
    #     if k not in common_bits:
    #         x += dict_prob_qc2[k]

    # print(x, theta)