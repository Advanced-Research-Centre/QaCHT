from qiskit import QuantumCircuit, Aer, execute
import numpy as np
from itertools import combinations
from math import ceil, log2

###############################################################################################

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

def setpartition_to_list(partition):
    """
    Input:
    ------
    partition: A permutation of causally connected qubits in form of sets.
    
    Output:
    ------
    A conversion to list of the set of permutations. 
    """
    qubit_partitions = []
    for el in list(partition):
        small_list = []
        for no in range(len(el)):
            small_list.append( list(el[no]) )
        qubit_partitions.append(small_list)
    return qubit_partitions

# for pairs in range(2,6):
#     pairings = setpartition_to_list(list(setpartition(list(range(0, 2*pairs)))))
#     print(pairs, len(pairings))

###############################################################################################

def causal_oracle(hypothesis_id = 0):
    """
    Output:
    -------
    The Causal Hypothesis Oracle
    """
    qc_C = QuantumCircuit(size_hypothesis*num_hypothesis)
    if hypothesis_id == 0:
        qc_C.id([0])
        qc_C.id([1])
    else:
        qc_C.swap([0],[1])
    gate = qc_C.to_gate(label = 'C_'+str(hypothesis_id))
    return gate

###############################################################################################

def controller_entangler(control_no):
    """
    Input:
    ------
    control_no = number of control qubits
    Output:
    -------
    Controlled Entangler Unitary.
    """
    qc_ent = QuantumCircuit(2)
    qc_ent.h([0])
    qc_ent.cx([0], [1])
    gate = qc_ent.to_gate(label = 'cEnt').control(control_no)
    return gate

def indexed_entangler(qc, control_qubits, target_qubits, bit):
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
        qc.append( controller_entangler(len(control_qubits)), total )
        for i in range(len(bit)):
            if bit[i] == '1':
                qc.x([control_qubits[0] + i])
    qc.barrier()
    return qc

###############################################################################################

def CHT(qb_hypothesis, pairings, hyp):
    """
    Input:
    ------
    qb_hypothesis = number of qubits in causal circuit (Bell connections),
    target_qubits = list of target qubits
    control_qubits = list of control qubits
    
    Output:
    -------
    Aritra dar causality hypothesis circuit
    """
    control_no = int( np.ceil( np.log2( qb_hypothesis ) ) )
    qc = QuantumCircuit( 2*qb_hypothesis + control_no )

    target_qubits_total = pairings
    control_qubits = []
    for el in range(control_no):
        control_qubits.append( 2*qb_hypothesis + el )
        qc.h([2*qb_hypothesis + el])
    qc.barrier()

    num = 0
    t = target_qubits_total[0][0]
    for i in range(len(target_qubits_total)):
        if t == target_qubits_total[i][0]:
            bit = format(num, f'0{control_no}b')
            indexed_entangler(qc, control_qubits, target_qubits_total[i], bit)
        else:
            t = target_qubits_total[i][0]
            num +=1
            bit = format(num, f'0{control_no}b')
            indexed_entangler(qc, control_qubits, target_qubits_total[i], bit)
    
    c_oracle = causal_oracle(hyp)
    for causal_q in range(qb_hypothesis):
        qc.append( c_oracle, [ causal_q, causal_q + qb_hypothesis ] )
    qc.barrier()
    return qc

###############################################################################################

def CHT_test(CHT_circuit):
    """
    Input:
    ------
    CHT_circuit = Is basically the causally connected circuit which is decomposed into A,B, and C subsystems. 

    Output:
    -------
    CHT_test = Is basically the statevector from the causally connected circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(CHT_circuit, simulator).result()
    return result.get_statevector(CHT_circuit)

###############################################################################################

if __name__ == "__main__":
    
    num_hypothesis = 2
    size_hypothesis = 1
    num_pairs = 2
    qb_hypothesis = size_hypothesis*(2*num_pairs)

    pairings = setpartition_to_list(list(setpartition(list(range(0, qb_hypothesis)))))
    # print(pairings)

    num_qubits = num_hypothesis*qb_hypothesis + ceil(log2(len(pairings)))
    # print(num_qubits)

    for hyp in range(0,num_hypothesis):
        CHT_circuit = CHT(qb_hypothesis, pairings, hyp)
        print("Hypothesis ",hyp)
        print(CHT_circuit)
        sv = CHT_test(CHT_circuit)
        print("State Vector ")
        for i in range(0,len(sv)):
            p = abs(sv[i])**2
            if p > 1e-5:
                print(bin(i)[2:].zfill(10),p)

###############################################################################################

'''
Hypothesis  0
           ░ ┌───────┐          ░      ┌───────┐                         ░      ┌───────┐                         ░ ┌──────┐                         ░
q_0: ──────░─┤0      ├──────────░──────┤0      ├─────────────────────────░──────┤0      ├─────────────────────────░─┤0     ├─────────────────────────░─
           ░ │  cEnt │          ░      │       │          ┌───────┐      ░      │       │          ┌───────┐      ░ │      │┌──────┐                 ░
q_1: ──────░─┤1      ├──────────░──────┤  cEnt ├──────────┤0      ├──────░──────┤       ├──────────┤0      ├──────░─┤      ├┤0     ├─────────────────░─
           ░ └──┬────┘┌───────┐ ░      │       │          │       │      ░      │  cEnt │          │  cEnt │      ░ │      ││      │┌──────┐         ░
q_2: ──────░────┼─────┤0      ├─░──────┤1      ├──────────┤  cEnt ├──────░──────┤       ├──────────┤1      ├──────░─┤  C_0 ├┤      ├┤0     ├─────────░─
           ░    │     │  cEnt │ ░      └──┬────┘          │       │      ░      │       │          └──┬────┘      ░ │      ││      ││      │┌──────┐ ░
q_3: ──────░────┼─────┤1      ├─░─────────┼───────────────┤1      ├──────░──────┤1      ├─────────────┼───────────░─┤      ├┤  C_0 ├┤      ├┤0     ├─░─
           ░    │     └──┬────┘ ░         │               └──┬────┘      ░      └──┬────┘             │           ░ │      ││      ││      ││      │ ░
q_4: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─┤1     ├┤      ├┤  C_0 ├┤      ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░ └──────┘│      ││      ││      │ ░
q_5: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────┤1     ├┤      ├┤  C_0 ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░         └──────┘│      ││      │ ░
q_6: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────────────┤1     ├┤      ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░                 └──────┘│      │ ░
q_7: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────────────────────┤1     ├─░─
     ┌───┐ ░    ││       ││     ░         ││                 ││          ░ ┌───┐   ││    ┌───┐┌───┐   ││    ┌───┐ ░                         └──────┘ ░
q_8: ┤ H ├─░────┼■───────┼■─────░─────────┼■─────────────────┼■──────────░─┤ X ├───┼■────┤ X ├┤ X ├───┼■────┤ X ├─░──────────────────────────────────░─
     ├───┤ ░    ││       ││     ░ ┌───┐   ││    ┌───┐┌───┐   ││    ┌───┐ ░ └───┘   ││    └───┘└───┘   ││    └───┘ ░                                  ░
q_9: ┤ H ├─░────┼■───────┼■─────░─┤ X ├───┼■────┤ X ├┤ X ├───┼■────┤ X ├─░─────────┼■─────────────────┼■──────────░──────────────────────────────────░─
     └───┘ ░    │        │      ░ └───┘   │     └───┘└───┘   │     └───┘ ░         │                  │           ░                                  ░
State Vector
0000000000 0.25000000000000044
0100000000 0.06250000000000008
0100000101 0.06250000000000017
0100001010 0.06250000000000022
0100001111 0.06250000000000022
1000000000 0.06250000000000008
1000000110 0.06250000000000011
1000001001 0.06250000000000008
1000001111 0.06250000000000011
1100000000 0.06250000000000022
1100000011 0.0625000000000003
1100001100 0.06250000000000022
1100001111 0.06250000000000028
Hypothesis  1
           ░ ┌───────┐          ░      ┌───────┐                         ░      ┌───────┐                         ░ ┌──────┐                         ░
q_0: ──────░─┤0      ├──────────░──────┤0      ├─────────────────────────░──────┤0      ├─────────────────────────░─┤0     ├─────────────────────────░─
           ░ │  cEnt │          ░      │       │          ┌───────┐      ░      │       │          ┌───────┐      ░ │      │┌──────┐                 ░
q_1: ──────░─┤1      ├──────────░──────┤  cEnt ├──────────┤0      ├──────░──────┤       ├──────────┤0      ├──────░─┤      ├┤0     ├─────────────────░─
           ░ └──┬────┘┌───────┐ ░      │       │          │       │      ░      │  cEnt │          │  cEnt │      ░ │      ││      │┌──────┐         ░
q_2: ──────░────┼─────┤0      ├─░──────┤1      ├──────────┤  cEnt ├──────░──────┤       ├──────────┤1      ├──────░─┤  C_1 ├┤      ├┤0     ├─────────░─
           ░    │     │  cEnt │ ░      └──┬────┘          │       │      ░      │       │          └──┬────┘      ░ │      ││      ││      │┌──────┐ ░
q_3: ──────░────┼─────┤1      ├─░─────────┼───────────────┤1      ├──────░──────┤1      ├─────────────┼───────────░─┤      ├┤  C_1 ├┤      ├┤0     ├─░─
           ░    │     └──┬────┘ ░         │               └──┬────┘      ░      └──┬────┘             │           ░ │      ││      ││      ││      │ ░
q_4: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─┤1     ├┤      ├┤  C_1 ├┤      ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░ └──────┘│      ││      ││      │ ░
q_5: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────┤1     ├┤      ├┤  C_1 ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░         └──────┘│      ││      │ ░
q_6: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────────────┤1     ├┤      ├─░─
           ░    │        │      ░         │                  │           ░         │                  │           ░                 └──────┘│      │ ░
q_7: ──────░────┼────────┼──────░─────────┼──────────────────┼───────────░─────────┼──────────────────┼───────────░─────────────────────────┤1     ├─░─
     ┌───┐ ░    ││       ││     ░         ││                 ││          ░ ┌───┐   ││    ┌───┐┌───┐   ││    ┌───┐ ░                         └──────┘ ░
q_8: ┤ H ├─░────┼■───────┼■─────░─────────┼■─────────────────┼■──────────░─┤ X ├───┼■────┤ X ├┤ X ├───┼■────┤ X ├─░──────────────────────────────────░─
     ├───┤ ░    ││       ││     ░ ┌───┐   ││    ┌───┐┌───┐   ││    ┌───┐ ░ └───┘   ││    └───┘└───┘   ││    └───┘ ░                                  ░
q_9: ┤ H ├─░────┼■───────┼■─────░─┤ X ├───┼■────┤ X ├┤ X ├───┼■────┤ X ├─░─────────┼■─────────────────┼■──────────░──────────────────────────────────░─
     └───┘ ░    │        │      ░ └───┘   │     └───┘└───┘   │     └───┘ ░         │                  │           ░                                  ░
State Vector
0000000000 0.25000000000000044
0100000000 0.06250000000000008
0101010000 0.06250000000000017
0110100000 0.06250000000000022
0111110000 0.06250000000000022
1000000000 0.06250000000000008
1001100000 0.06250000000000011
1010010000 0.06250000000000008
1011110000 0.06250000000000011
1100000000 0.06250000000000022
1100110000 0.0625000000000003
1111000000 0.06250000000000022
1111110000 0.06250000000000028
'''

###############################################################################################

'''
    - measure only A (Z-basis)
    - check measure statistics 
    - compare between swap/identity
'''