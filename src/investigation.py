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

# print((3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2))))
# exit()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
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

def oracle_distance():
    fig, ax1 = plt.subplots(1, 1, figsize=(4.7,4) )
    qc_id = oracle_type(0, 'id')
    qc_id_unitary = Operator(qc_id).data
    mxd_choi_mat = np.eye(len(qc_id_unitary))/len(qc_id_unitary)
    theta_range = np.arange(0, 4*np.pi, np.pi/10)
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
    ax1.set_ylabel("$ \\Delta [U_{\\small\\textrm{orc}}^{H_0}(0,0), U_{\\small\\textrm{orc}}^{H_1}(\\pi,\\theta_y)] $")
    ax1.set_xlabel("$\\theta_y$")
    ax1.legend(loc = 'center', ncol = 2, bbox_to_anchor=(0.5, 1.12), fontsize = 10)
    fig.tight_layout()
    plt.savefig('plot/diff_process_dist.pdf')
    plt.savefig('plot/diff_process_dist.png')

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        self.set_some_features_visibility(False)
        super(MyAxes3D, self).draw(renderer)
        self.set_some_features_visibility(True)
        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        zaxis.axes._draw_grid = False
        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes
        zaxis.axes._draw_grid = draw_grid_old

def practical_case_error_prob(ax1, ax2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,5) )
    theta_range = np.arange(0, 4*np.pi, np.pi/3)
    theta_x_list = theta_range
    theta_oracle_list = theta_range
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]

    if hypothesis == "identity":
        theta_oracle_list = [0.0]
        theta_x_list = [0.0]
    elif hypothesis == "swap-ry":
        theta_oracle_list = np.arange(0, 4*np.pi, np.pi/20)
        theta_x_list = np.arange(0, 4*np.pi, np.pi/20)
    
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
    
    hypothesis_list = ["identity", "swap-ry"]
    hypothesis = hypothesis_list[1]
    theta_oracle_list, theta_x_list = np.arange(0, 4*np.pi, np.pi/10), np.arange(0, 4*np.pi, np.pi/10)
    sx, sy = theta_oracle_list.size, theta_x_list.size
    theta_oracle_list_plot, theta_x_list_plot = numpy.tile(theta_oracle_list, (sy, 1)), numpy.tile(theta_x_list, (sx, 1)).T
    prac_prob_plot = np.zeros((len(theta_oracle_list), len(theta_x_list)))

    for no, theta_oracle in enumerate(theta_oracle_list):
        U = []
        for theta_x in theta_x_list:
            theta_oracle, theta_x = round(theta_oracle, 3), round(theta_x, 3)
            prob = 1- distinguishing_probability(hypothesis, 'had', theta_oracle, theta_x)
            U.append(prob)
        prac_prob_plot[no] = U
    
    _, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4.6),subplot_kw=dict(projection='3d'))
    norm = plt.Normalize( theta_oracle_list.min(), theta_oracle_list.max() )
    colors = cm.viridis( norm(theta_oracle_list_plot) )
    rcount, ccount, _ = colors.shape
    
    # theta_x
    surf = ax1.plot_surface(theta_oracle_list_plot, theta_x_list_plot, prac_prob_plot, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(np.pi)) # theta_y np.pi/2
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2)) # theta_x np.pi/2
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1.set_xlabel('$\\theta_y$', labelpad=10)
    ax1.set_ylabel('$\\theta_x$', labelpad=10)
    ax1.set_zlabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{prac}}$', labelpad=10)
    ax1.view_init(elev=None, azim=15) # theta_x
    
    # theta_y
    surf = ax2.plot_surface(theta_oracle_list_plot, theta_x_list_plot, prac_prob_plot, rcount=rcount, ccount=ccount,
                       facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2)) # theta_y np.pi/2
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi)) # theta_x np.pi/2
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax2.set_xlabel('$\\theta_y$', labelpad=10)
    ax2.set_ylabel('$\\theta_x$', labelpad=10)
    ax2.set_zlabel('$p_{\\small\\textrm{err}}^{\\small\\textrm{prac}}$', labelpad=10)
    ax2.view_init(40, azim=75)
    # ax2.zaxis.set_label_position('top')
    # plt.show()


#data/dict_prob_initial_ora_identity_ang_0.0_oracle_ang_0.0_theta_x_0.0_initial_initialization_had.p
if __name__ == "__main__":

    oracle_distance()

    drawPropagation()