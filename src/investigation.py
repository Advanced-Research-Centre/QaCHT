from cProfile import label
from cmath import log
from turtle import color
from main_circuit import *
from matplotlib import cm
import matplotlib.pyplot as plt
from qiskit.quantum_info.operators import Operator, process_fidelity
from qiskit.quantum_info.operators import Chi, Choi
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# # Grab some test data.
# X, Y, Z = axes3d.get_test_data(0.05)
# print(X.shape)
# print(X.shape)
# print(X.shape)
# # Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
# plt.show()

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


def distinguishing_probability(hypothesis, gate, theta_oracle, theta_x):
    file_hypo_1 = f'data/dict_prob_initial_hypo_identity_oracle_ang_0.0_theta_x_0.0_initial_initialization_{gate}.p'
    file_hypo_2 = f'data/dict_prob_initial_hypo_{hypothesis}_oracle_ang_{theta_oracle}_theta_x_{theta_x}_initial_initialization_{gate}.p'
    dict_hypo_1 = pickle.load(open(file_hypo_1, "rb"))
    dict_hypo_2 = pickle.load(open(file_hypo_2, "rb"))

    list_key_hypo_1 = []
    list_key_hypo_2 = []
    for k in dict_hypo_1.keys():
        list_key_hypo_1.append(k)
    for k in dict_hypo_2.keys():
        list_key_hypo_2.append(k)

    common_bits = []
    for i in list_key_hypo_1:
        for j in list_key_hypo_2:
            if i == j:
                common_bits.append(i)

    x = 0
    for k in list_key_hypo_2:
        if k not in common_bits:
            # print(k)
            # if k not in list_key_identity:
            x += dict_hypo_2[k]
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
    if type == 'ry-swap':
        qc.ry(theta, [1])
        qc.swap([1], [0])
    else:
        qc.id([0])
        qc.id([1])
    
    return qc

def DeltaT(dm_i, dm_j):
    dist = np.real(0.5* np.trace( np.sqrt((dm_i - dm_j)**2 )))
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
    theta_range = np.arange(0, 4*np.pi, np.pi/5)
    d1,d2,d3 = [],[],[]
    # prob = [] #HACK
    for t in theta_range:
        qc_cry = oracle_type(t, 'ry-swap')
        cry_choi_op = Operator(qc_cry).data
        dist1 = DeltaT(mxd_choi_mat,cry_choi_op)
        dist2 = DeltaB(mxd_choi_mat,cry_choi_op)
        dist3 = DeltaHS(mxd_choi_mat,cry_choi_op)
        d1.append(dist1)
        d2.append(dist2)
        d3.append(dist3)
        # prob.append(dist1*0.60205999132) #HACK
    
    ax1.plot(theta_range, d1, 'r--v', markerfacecolor='none', label = "Trace distance")
    ax1.plot(theta_range, d2, 'k-', label = "Bures distance")
    ax1.plot(theta_range, d3, 'g--o', markerfacecolor='none', label = "Hilbert-Schmidt distance")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    # ax2.plot(prob) #HACK
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_ylabel("$ \\Delta\\left[CRY(\\theta), \mathbb{I}\\otimes\mathbb{I}\\right] $")
    ax1.set_xlabel("$\\theta$")
    ax1.legend(loc = 'center', ncol = 2, bbox_to_anchor=(0.5, 1.1))


def practical_case_error_prob(ax1, ax2):

    theta_range = np.arange(0, 4*np.pi, np.pi/3)
    theta_x_list = theta_range
    theta_oracle_list = theta_range
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]


    if hypothesis == "identity":
        theta_oracle_list = [0.0]
        theta_x_list = [0.0]
    elif hypothesis == "swap-ry":
        theta_oracle_list = np.arange(0, 4*np.pi, np.pi/10)
        theta_x_list = np.arange(0, 4*np.pi, np.pi/10)
    
    
    for theta_oracle in theta_oracle_list:
        list_data = []
        list_angle = []
        for theta_x in theta_x_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            list_angle.append(theta_x)
            list_data.append(prob)
        ax1.plot(list_angle, list_data)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    ax1.set_xlabel('$\\theta_x$')
    ax1.set_ylabel('$P_{err}$')


    for theta_x in theta_x_list:
        list_data = []
        list_angle = []
        for theta_oracle in theta_oracle_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            list_angle.append(theta_oracle)
            list_data.append(prob)
        ax2.plot(list_angle, list_data)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax2.set_xlabel('$\\theta_{y}$')

def drawPropagation():
    """ beta2 in ps / km
        C is chirp
        z is an array of z positions """
    
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]

    T, z = np.arange(0, 4*np.pi, np.pi/10), np.arange(0, 4*np.pi, np.pi/10)
    sx, sy = T.size, z.size
    T_plot, z_plot = numpy.tile(T, (sy, 1)), numpy.tile(z, (sx, 1)).T
    U_plot = np.zeros((len(T), len(T)))

    for no, theta_oracle in enumerate(T):
        U = []
        for theta_x in z:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            U.append(prob)
        U_plot[no] = U
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    norm = plt.Normalize( T_plot.min(), T_plot.max() )
    colors = cm.viridis( norm(T_plot) )
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(T_plot, z_plot, U_plot, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel('$\\theta_y$')
    ax.set_ylabel('$\\theta_x$')
    ax.set_zlabel('$P_{err}$')

#data/dict_prob_initial_ora_identity_ang_0.0_oracle_ang_0.0_theta_x_0.0_initial_initialization_had.p
if __name__ == "__main__":

    drawPropagation()
    plt.savefig( f'plot/practical_case_error_prob.pdf' )
    plt.savefig( f'plot/practical_case_error_prob.png' )
    exit()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5), sharey=True )
    # operations_vs_linearly_independent_state(ax1)
    # operations_vs_subsystem_dim(ax2)
    # plt.tight_layout()
    # plt.savefig( f'plot/number_of_operations.pdf' )
    # plt.savefig( f'plot/number_of_operations.png' )



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5), sharey=True )
    # ax1.legend()
    ax = fig.add_subplot(projection='3d')
    practical_case_error_prob(ax)
    plt.savefig( f'plot/practical_case_error_prob.pdf' )
    plt.savefig( f'plot/practical_case_error_prob.png' )
    exit()
    # oracle_distance(ax1)
    # plt.savefig( f'plot/plot.pdf' )
    # plt.savefig( f'plot/plot.png' )
    # plt.show()
    # exit()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5) )
    ax1.set_ylim(-0.05,0.05)
    practical_case_error_prob(ax1)
    plt.show()
    # exit()
    oracle_distance(ax2)
    plt.tight_layout()
    plt.savefig( f'plot/prac_error_prob.pdf' )
    plt.savefig( f'plot/prac_error_prob.png' )
    exit()
    plt.savefig( f'plot/limiting_error_prob.pdf' )
    plt.savefig( f'plot/limiting_error_prob.png' )
    plt.show()