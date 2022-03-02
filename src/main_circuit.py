from qiskit import *
from qiskit import Aer
import numpy as np


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


def aritra_dar_causality( aritra_dar_dimension ):
    """
    
    aritra_dar_dimension = number of qubits in causal circuit (Bell connections),
    target_qubits = list of target qubits
    control_qubits = list of control qubits

    """
    qc = QuantumCircuit( aritra_dar_dimension + np.ceil( np.log2( aritra_dar_dimension ) ) )

    target_qubits_total = [ [ 0, 1 ], [ 2, 3 ] , [ 0, 2 ], [ 1, 3 ], [ 0, 3 ], [ 1, 2 ] ] #TODO: Tamal dar karshaji
    
    control_qubits = []
    for el in range(aritra_dar_dimension//2):
        control_qubits.append( aritra_dar_dimension + el )

    for target_qubits in target_qubits_total:
        total = control_qubits + target_qubits
        qc.append( cbell(), total )
        qc.barrier()
    
    return qc


print(aritra_dar_causality(4))

qc = QuantumCircuit(4)

# qc.cent()