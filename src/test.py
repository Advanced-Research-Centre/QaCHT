from qiskit import *
import numpy

qc1, qc2 = QuantumCircuit(4), QuantumCircuit(4)
qc2.h(0)
qc2.h(1)
qc2.h(2)
qc2.h(3)

dict_prob_qc1 = {}
dict_prob_qc2 = {}

simulator = Aer.get_backend('statevector_simulator')
result_qc1 = execute(qc1, simulator).result()
statevector_qc1 = result_qc1.get_statevector( qc1 ).data

result_qc2 = execute(qc2, simulator).result()
statevector_qc2 = result_qc2.get_statevector( qc2 ).data

print(statevector_qc1)
print('-------------------')
print(statevector_qc2)

for i in range(len(statevector_qc1)):
    p = abs(statevector_qc1[i])
    if p > 1e-5:
        dict_prob_qc1[f'{bin(i)[2:].zfill( 4 )}'] = p**2

for i in range(len(statevector_qc2)):
    p = abs(statevector_qc2[i])
    if p > 1e-5:
        dict_prob_qc2[f'{bin(i)[2:].zfill( 4 )}'] = p**2

list_key_hypo_1 = []
list_key_hypo_2 = []
for k in dict_prob_qc1.keys():
    list_key_hypo_1.append(k)
for k in dict_prob_qc2.keys():
    list_key_hypo_2.append(k)


common_bits = []
for i in list_key_hypo_1:
    for j in list_key_hypo_2:
        if i == j:
            common_bits.append(i)
x = 0
for k in list_key_hypo_2:
    if k not in common_bits:
        x += dict_prob_qc2[k]

print(1-x)