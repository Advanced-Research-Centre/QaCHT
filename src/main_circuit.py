from qiskit import *
from qiskit import Aer
import numpy as np
from itertools import combinations


def setpartition(iterable, n=2):
    iterable = list(iterable)
    partitions = combinations(combinations(iterable, n), r=int(len(iterable)/n))
    for partition in partitions:
        seen = set()
        for group in partition:
            if seen.intersection(group):
                break
            seen.update(group)
        else:
            yield partition


def setpartition_to_list(setpartition):
    qubit_partitions = []
    for el in list(setpartition):
        small_list = []
        for no in range(len(el)):
            small_list.append( list(el[no]) )
        qubit_partitions.append(small_list)
    return qubit_partitions


def cbell():
    """
    return Bell's Unitary.
    """
    n_ent = 2
    qc_ent = QuantumCircuit(n_ent)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'Max. Ent. gate').control(2)
    return gate


def aritra_dar_causality( aritra_dar_dimension , qubit_partitions ):
    """
    
    aritra_dar_dimension = number of qubits in causal circuit (Bell connections),
    target_qubits = list of target qubits
    control_qubits = list of control qubits

    """
    target = np.ceil( np.log2( aritra_dar_dimension ) )
    qc = QuantumCircuit( aritra_dar_dimension + target )
    target_qubits_total = qubit_partitions #TODO: Tamal dar karshaji
    control_qubits = []

    for el in range(aritra_dar_dimension//2):
        control_qubits.append( aritra_dar_dimension + el )
    
    for target_no, target_qubits in enumerate(target_qubits_total):
        for target in target_qubits:
            total = control_qubits + target
            qc.append( cbell(), total )
            qc.barrier()
    
    return qc


if __name__ == "__main__":
    aritra_dar_dimension = 4
    partition = list(setpartition(list(range(aritra_dar_dimension))))
    qubit_partitions = setpartition_to_list(partition)
    circuit = aritra_dar_causality( aritra_dar_dimension , qubit_partitions )
    print(circuit)