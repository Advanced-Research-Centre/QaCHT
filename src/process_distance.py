
from cirq import fidelity
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Choi
from qiskit.quantum_info import process_fidelity, SuperOp, Operator, state_fidelity, DensityMatrix
import matplotlib.pyplot as plt
# import scipy.linalg as la
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
# csfont = {'fontname':'Sans Serif'}
# hfont = {'fontname':'Helvetica'}


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

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
    
    # diff_dag = diff.conjugate().transpose()
    # dist = np.real(0.5* np.trace(la.sqrtm(np.matmul(diff_dag, diff))))
    dist = np.real(0.5* np.trace( np.abs(diff) ))
    return dist

def DeltaB(dm_i, dm_j):

    fid = process_fidelity(Operator(dm_j), Operator(dm_i))
    dist = 2*(1-np.sqrt(fid))
    return dist

def DeltaHS(dm_i, dm_j):

    dist = np.abs(np.trace((dm_i - dm_j)**2))
    return dist

if __name__ == "__main__":
    qc_id = oracle_type(0, 'id')
    qc_id_unitary = Operator(qc_id).data
    # print(qc_id_unitary)
    # id_choi_mat = Choi(qc_id).data
    mxd_choi_mat = np.eye(len(qc_id_unitary))/len(qc_id_unitary)
    # print(type(mxd_choi_mat), type(id_choi_mat))
    theta_range = np.arange(0, 8*np.pi, 0.2)
    d1 = []
    d2 = []
    d3 = []
    x = np.linspace(0, 8*np.pi)
    for t in theta_range:
        qc_cry = oracle_type(t, 'cry')
        cry_choi_op = Operator(qc_cry).data
        dist1 = DeltaT(mxd_choi_mat,cry_choi_op)
        dist2 = DeltaB(mxd_choi_mat,cry_choi_op)
        dist3 = DeltaHS(mxd_choi_mat,cry_choi_op)
        d1.append(dist1)
        d2.append(dist2)
        d3.append(dist3)
    
    fig, ax = plt.subplots( figsize=(5,4) )
    ax.plot(theta_range, d1, 'r--v', markerfacecolor='none', label = "Trace distance")
    ax.plot(theta_range, d2, 'k-', label = "Bures distance")
    ax.plot(theta_range, d3, 'g--o', markerfacecolor='none', label = "Hilbert-Schmidt distance")
    # plt.ylabel("Process Distance of Choi CRY(theta) and Choi IxI")
    # plt.ylim([0, np.ceil(max(d3))+0.05])
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_ylabel("$ \\Delta\\left[CRY(\\theta), \mathbb{I}\\otimes\mathbb{I}\\right] $")
    ax.set_xlabel("$\\theta$")
    ax.legend(loc = 'upper center')
    fig.tight_layout()
    plt.savefig('plot/diff_process_dist.pdf')
    plt.savefig('plot/diff_process_dist.png')
    plt.show()