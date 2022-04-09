from qiskit import *
from qiskit import Aer
import numpy as np
from itertools import combinations
import pickle

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
    """
    Input:
    ------
    setpartition: A permutation of causally connected qubits in form of sets.
    
    Output:
    A conversion to list of the set of permutations. 

    """
    qubit_partitions = []
    for el in list(setpartition):
        small_list = []
        for no in range(len(el)):
            small_list.append( list(el[no]) )
        qubit_partitions.append(small_list)
    return qubit_partitions

def circuit_subroutine(qc, control_qubits, target_qubits, bit):
    """
    Inputs:
    -------
    qc = Quantum circuit.
    control_qubits = list of control qubits.
    target_qubits = list of target qubits.
    bit = bit value corresponding to the permutations i.e. for 0: '00'
    
    Outputs:
    --------
    Aritra dar causality hypothesis circuit 

    """
    for target in target_qubits:
        total = control_qubits + target
        for i in range(len(bit)):
            if bit[i] == '1':
                qc.x([control_qubits[0] + i])
        qc.append( cbell(len(control_qubits)), total )
        for i in range(len(bit)):
            if bit[i] == '1':
                qc.x([control_qubits[0] + i])
    qc.barrier()
    return qc


def cbell(control_no):
    """
    Input:
    ------
    control_no = number of control qubits
    Output:
    -------
    Controlled Bell's Unitary.
    """
    qc_ent = QuantumCircuit(2)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'C-Bell Unitary').control(control_no)
    return gate

def bell():
    """
    Output:
    -------
    Bell's Unitary.
    """
    qc_ent = QuantumCircuit(2)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'Bell Unitary')
    return gate


def causal_oracle(opt):
    """
    Output:
    -------
    Bell's Unitary.
    """
    qc_ent = QuantumCircuit(2)
    if opt == 'swap':
        qc_ent.swap([0],[1])
    elif opt == 'identity':
        qc_ent.id([0])
        qc_ent.id([1])
    gate = qc_ent.to_gate(label = 'C')
    return gate


def aritra_dar_causality( aritra_dar_dimension , qubit_partitions, opt, theta, gate ):
    """
    Input:
    ------
    aritra_dar_dimension = number of qubits in causal circuit (Bell connections),
    target_qubits = list of target qubits
    control_qubits = list of control qubits
    
    Output:
    -------
    Aritra dar causality hypothesis circuit
    """
    control_no = int( np.ceil( np.log2( aritra_dar_dimension ) ) )
    qc = QuantumCircuit( 2*aritra_dar_dimension + control_no )

    target_qubits_total = qubit_partitions
    control_qubits = []
    for el in range(control_no):
        control_qubits.append( 2*aritra_dar_dimension + el )
        qc.h([2*aritra_dar_dimension+el])
    
    for q_rot in range(aritra_dar_dimension):
        if gate == 'rx':
            qc.rx( theta, q_rot )
        elif gate == 'ry':
            qc.ry( theta, q_rot )
        elif gate == 'rz':
            qc.rz( theta, q_rot )

    num = 0
    t = target_qubits_total[0][0]
    for i in range(len(target_qubits_total)):
        if t == target_qubits_total[i][0]:
            bit = format(num, f'0{control_no}b')
            circuit_subroutine(qc, control_qubits, target_qubits_total[i], bit)
        else:
            t = target_qubits_total[i][0]
            num +=1
            bit = format(num, f'0{control_no}b')
            circuit_subroutine(qc, control_qubits, target_qubits_total[i], bit)
    
    # for _ in range(oracle_repeation):
    c_oracle = causal_oracle(opt)
    for causal_q in range(aritra_dar_dimension):
        qc.append( c_oracle, [ causal_q, causal_q + aritra_dar_dimension ] )
    
    return qc

def aritra_dar_dosha( aritra_der_bortoni ):
    """
    Input:
    ------
    aritra_der_bortoni = Is besically the causally connected circuit which is decomposed into
    A,B, and C subsystems. "C" subsystem is connect to subsystem "B" with 2-controlled Bell's Unitary
    and the subsystem "A" is connected to the subsystem "B" by Bell's Unitaries.

    Output:
    -------
    aritra_der_dosha = Is basically the statevector from the causally connected circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(aritra_der_bortoni, simulator).result()
    return result.get_statevector( aritra_der_bortoni )

if __name__ == "__main__":
    

    aritra_dar_dimension = 4
    list_opts = [ 'swap', 'identity' ]
    total_qubit_required = 2*aritra_dar_dimension + int( np.ceil( np.log2( aritra_dar_dimension ) ) )
    partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
    qubit_partitions = setpartition_to_list(partition)
    gate = 'ry'

    for theta in np.arange(0, 2*np.pi, 0.02):
        for opt in list_opts:
            dict_prob = {}
            print(f'{opt} before output for angle {theta}')
            print('--------------')
            aritra_dar_bortoni = aritra_dar_causality( aritra_dar_dimension , qubit_partitions, opt, theta, gate )
            print(aritra_dar_bortoni)
            exit()
            aritra_chiribella_dosha = aritra_dar_dosha(  aritra_dar_bortoni ) 
            for i in range(len(aritra_chiribella_dosha)):
                p = abs(aritra_chiribella_dosha[i])
                if p > 1e-5:
                    dict_prob[f'{bin(i)[2:].zfill( total_qubit_required )}'] = p**2
            file = f'data/{opt}_dict_prob_{theta}_initialization_{gate}.p'
            with open(file, 'wb') as handle:
                pickle.dump(dict_prob, handle)

# SWAP
    # 00 0000 0000 0.5000000000000004
    # 01 0000 0000 0.25000000000000017
    # 01 0000 0101 0.25000000000000033
    # 01 0000 1010 0.25000000000000044
    # 01 0000 1111 0.25000000000000044
    # 10 0000 0000 0.25000000000000017
    # 10 0000 0110 0.2500000000000002
    # 10 0000 1001 0.25000000000000017
    # 10 0000 1111 0.2500000000000002
    # 11 0000 0000 0.25000000000000044
    # 11 0000 0011 0.2500000000000006
    # 11 0000 1100 0.25000000000000044
    # 11 0000 1111 0.25000000000000056
# IDENTITY
    # 00 0000 0000 0.5000000000000004
    # 01 0000 0000 0.25000000000000017
    # 01 0101 0000 0.25000000000000033
    # 01 1010 0000 0.25000000000000044
    # 01 1111 0000 0.25000000000000044
    # 10 0000 0000 0.25000000000000017
    # 10 0110 0000 0.2500000000000002
    # 10 1001 0000 0.25000000000000017
    # 10 1111 0000 0.2500000000000002
    # 11 0000 0000 0.25000000000000044
    # 11 0011 0000 0.2500000000000006
    # 11 1100 0000 0.25000000000000044
    # 11 1111 0000 0.25000000000000056

'''
    TODO------
    - measure only A (Z-basis)
    - check measure statistics 
    - compare between swap/identity
'''