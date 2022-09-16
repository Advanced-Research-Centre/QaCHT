from cProfile import label
from cmath import log
from turtle import color
from main_circuit import *
import matplotlib.pyplot as plt
from qiskit.quantum_info.operators import Operator, process_fidelity
from qiskit.quantum_info.operators import Chi, Choi
from matplotlib import rcParams
import matplotlib.font_manager as font_manager




plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
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


def distinguishing_probability(list_opts, gate, theta_init, theta_oracle):
    dict_swap = pickle.load(open(f'data/{list_opts[0]}_dict_prob_initial_ang_{theta_init}_oracle_ang_{theta_oracle}_initial_initialization_{gate}.p', "rb"))
    dict_identity = pickle.load(open(f'data/{list_opts[1]}_dict_prob_initial_ang_{theta_init}_oracle_ang_{theta_oracle}_initial_initialization_{gate}.p', "rb"))

    list_key_swap = []
    list_key_identity = []
    for k in dict_swap.keys():
        list_key_swap.append(k)
    for k in dict_identity.keys():
        list_key_identity.append(k)

    common_bits = []
    for i in list_key_swap:
        for j in list_key_identity:
            if i == j:
                common_bits.append(i)

    x = 0
    for k in list_key_swap:
        if k not in common_bits:
            # if k not in list_key_identity:
            x += dict_swap[k]
            # else:
                # x += dict_identity[k]
    return x

def operations_vs_linearly_independent_state(ax1):
    subsystem_sim_list = []
    partition_list = []
    for aritra_dar_dimension in range(2, 12):
        partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
        # subsystem_sim_list.append(np.ceil(np.log(len(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))))
        subsystem_sim_list.append(len(partition))
        partition_list.append(len(partition)*len(partition[0]))
    ax1.plot( subsystem_sim_list, partition_list, '-x' )
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.semilogy( [3], [6], 'ro', alpha=.3, ms=9, lw=3)
    ax1.set_ylabel('Number of operations', fontsize = 14)
    ax1.set_xlabel('$r$', fontsize = 14)

def operations_vs_subsystem_dim(ax2):
    aritra_dar_dimension = 4
    subsystem_sim_list = []
    partition_list = []
    for aritra_dar_dimension in range(2, 12):
        partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
        subsystem_sim_list.append(aritra_dar_dimension)
        partition_list.append(len(partition)*len(partition[0]))
    ax2.plot( subsystem_sim_list, partition_list, '-x' )
    ax2.set_yscale('log')
    ax2.semilogy( [4], [6], 'ro', alpha=.3, ms=9, lw=3, label = 'Simulation presented in article')
    # ax2.set_ylabel('Number of operations', fontsize = 14)
    ax2.set_xlabel('$N_{A}$',fontsize = 14)
    ax2.legend()


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
    dist = np.real(0.5* np.trace( np.abs(diff) ))
    return dist

def DeltaB(dm_i, dm_j):
    fid = process_fidelity(Operator(dm_j), Operator(dm_i))
    dist = 2*(1-np.sqrt(fid))
    return dist

def DeltaHS(dm_i, dm_j):
    dist = np.abs(np.trace((dm_i - dm_j)**2))
    return dist

def oracle_distance(ax1):
    qc_id = oracle_type(0, 'id')
    qc_id_unitary = Operator(qc_id).data
    mxd_choi_mat = np.eye(len(qc_id_unitary))/len(qc_id_unitary)
    theta_range = np.arange(0, 8*np.pi, 0.2)
    d1,d2,d3 = [],[],[]
    for t in theta_range:
        qc_cry = oracle_type(t, 'cry')
        cry_choi_op = Operator(qc_cry).data
        dist1 = DeltaT(mxd_choi_mat,cry_choi_op)
        dist2 = DeltaB(mxd_choi_mat,cry_choi_op)
        dist3 = DeltaHS(mxd_choi_mat,cry_choi_op)
        d1.append(dist1)
        d2.append(dist2)
        d3.append(dist3)
    
    ax1.plot(theta_range, d1, 'r--v', markerfacecolor='none', label = "Trace distance")
    ax1.plot(theta_range, d2, 'k-', label = "Bures distance")
    ax1.plot(theta_range, d3, 'g--o', markerfacecolor='none', label = "Hilbert-Schmidt distance")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_ylabel("$ \\Delta\\left[CRY(\\theta), \mathbb{I}\\otimes\mathbb{I}\\right] $")
    ax1.set_xlabel("$\\theta$")
    ax1.legend(loc = 'center', ncol = 2, bbox_to_anchor=(0.5, 1.1))

def practical_case_error_prob(ax2):
    theta_range =  np.arange(0, 8*np.pi, 0.05)
    total_list_opts = [ [ 'cry', 'identity' ] ]
    calculate = 'prob'
    gate_list = 'had'
    if gate_list == 'had':
        theta_init_list = [0]
    else:
        theta_init_list = theta_range
    line_style = [ 'r--x', 'b--o' ]
    dict_error_prob = {}

    for no, list_opts in enumerate(total_list_opts):
        for l in list_opts:
            if l  == 'cry':
                theta_oracle_list = theta_range
            elif l == 'swap':
                theta_oracle_list = [0]

        for gate in [gate_list]:
            for theta_init in theta_init_list:
                list_data = []
                list_angle = []
                for theta_oracle in theta_oracle_list:
                    theta_init = round(theta_init, 2)
                    theta_oracle = round(theta_oracle, 2)
                    prob = 1 - distinguishing_probability(list_opts, gate, theta_init, theta_oracle)
                    list_angle.append(theta_oracle)
                    if calculate == 'rate':
                        rate = -log(prob)/4
                        list_data.append(rate)
                    elif calculate == 'prob':
                        list_data.append(prob)

                if no == 0:
                    ax2.plot( list_angle, list_data, 'r-')
                    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
                    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
                    ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
                else:
                    ax2.plot( list_angle, list_data, label = f'{gate}, oracle = [swap,id]' )
                ax2.set_xlabel( '$\\theta$' )
            
            dict_error_prob[f'{no}'] = list_data

    difference_list = []
    for el in dict_error_prob['0']:
        difference = el - dict_error_prob['0'][0]
        difference_list.append(difference)

    if calculate == 'rate':
        plt.plot( theta_range, [0.60205999132]*len(theta_range), 'k-', label = 'chiribella disrimination rate' )
        plt.ylabel('Discrimination rate', fontsize=14)
    
    elif calculate == 'prob':
        chiribella_error_prob = (3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2)))
        ax2.plot( theta_range, [chiribella_error_prob]*len(theta_range), 'k--', label = 'Limiting case error prob. ($p_{\\small \\textrm{err}}$)' )
        ax2.set_ylabel( '$p_{\\small \\textrm{err}}^{\\small \\textrm{prac}}$',  fontsize = 14 )
    ax2.legend(loc = 'center', bbox_to_anchor=(0.5, 1.06))

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5), sharey=True )
    operations_vs_linearly_independent_state(ax1)
    operations_vs_subsystem_dim(ax2)
    plt.tight_layout()
    plt.savefig( f'plot/number_of_operations.pdf' )
    plt.savefig( f'plot/number_of_operations.png' )


    exit()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5) )
    practical_case_error_prob(ax1)
    oracle_distance(ax2)
    plt.tight_layout()
    plt.savefig( f'plot/limiting_error_prob.pdf' )
    plt.savefig( f'plot/limiting_error_prob.png' )
    plt.show()