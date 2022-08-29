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
    "font.size": 10,
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

def number_of_perms():
    aritra_dar_dimension = 4
    subsystem_sim_list = []
    partition_list = []
    for aritra_dar_dimension in range(2, 12):
        partition = list(setpartition(list(range(aritra_dar_dimension, 2*aritra_dar_dimension))))
        subsystem_sim_list.append(aritra_dar_dimension)
        partition_list.append(len(partition))
    ax.semilogy( subsystem_sim_list, partition_list, '-x' )
    ax.semilogy( [4], [3], 'ro', alpha=.4, ms=9, lw=3, label = 'Simulation presented in article')
    ax.set_ylabel('Number of permutations')
    ax.set_xlabel('$N_A \ \\textrm{or} \ N_B$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/numb_of_perm.pdf')

if __name__ == "__main__":

    fig, ax = plt.subplots( figsize=(5,4) )
    theta_range =  np.arange(0, 4*np.pi, 0.05)
    total_list_opts = [ [ 'cry', 'identity' ] ] #, [ 'swap', 'identity' ] ]
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

        for gateno, gate in enumerate([gate_list]):
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
                    ax = plt.gca()
                    ax.plot( list_angle, list_data, line_style[no])#, label = f'{gate}', oracle = [cry,id]' )
                    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
                    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
                    # ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

                else:
                    ax.plot( list_angle, list_data, line_style[no], label = f'{gate}, oracle = [swap,id]' )
                ax.set_xlabel( '$\\theta$' )
            
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
        plt.plot( theta_range, [chiribella_error_prob]*len(theta_range), 'k--', label = 'Limiting case error prob. ($p_{\\small \\textrm{err}}$)' )
        ax.set_ylabel( '$p_{\\small \\textrm{err}}^{\\small \\textrm{prac}}$',  fontsize = 14 )

    plt.legend(loc = 'best')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlabel(fontsize = 13)
    plt.ylabel(fontsize = 13)

    plt.tight_layout()
    plt.savefig( f'plot/hypothesis_distinguishing_{calculate}.pdf' )
    plt.savefig( f'plot/hypothesis_distinguishing_{calculate}.png' )
    plt.show()