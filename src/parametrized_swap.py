from qiskit import QuantumCircuit
# from qiskit.circuit import Parameter
# import numpy as np
from qiskit import *

theta = 0.9

swap_circ = QuantumCircuit(2)
swap_circ.x([0])
swap_circ.swap(0,1)
swap_circ.measure_all()
simulator = BasicAer.get_backend('statevector_simulator')
results = execute(swap_circ, simulator).result()
swap_count = results.get_statevector()
abs_state = []
for i in swap_count:
    abs_state.append(abs(i))
print(abs_state)
print('-------')




circuit = QuantumCircuit(2)
circuit.x([0])
circuit.rxx(theta, 0, 1)
circuit.ryy(theta, 0, 1)
circuit.measure_all()
simulator = BasicAer.get_backend('statevector_simulator')
results = execute(circuit, simulator).result()
statevector = results.get_statevector()
abs_state = []
for i in statevector:
    abs_state.append(abs(i))
print(abs_state)


