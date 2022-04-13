from cProfile import label
from cmath import log
from main_circuit import *
import matplotlib.pyplot as plt

def distinguishing_probability(gate, theta_init, theta_oracle):
    list_opts = [ 'cry', 'identity' ]
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

if __name__ == "__main__":

    calculate = 'prob'
    gate_list = 'had' #, 'ry', 'rz' ]
    if gate_list == 'had':
        theta_init_list = [0]
    else:
        theta_init_list = np.arange(0, 2*np.pi, 0.5)
    
    theta_oracle_list = np.arange(0, 2*np.pi, 0.2)
    line_opt = [ 'r--x', 'b--o', 'g--' ]
    for gateno, gate in enumerate([gate_list]):
        for theta_init in theta_init_list:
            list_data = []
            list_angle = []
            for theta_oracle in theta_oracle_list:
                theta_init = round(theta_init, 2)
                theta_oracle = round(theta_oracle, 2)
                prob = 1 - distinguishing_probability(gate, theta_init, theta_oracle)
                list_angle.append(theta_oracle)
                if calculate == 'rate':
                    rate = -log(prob)/4
                    list_data.append(rate)
                elif calculate == 'prob':
                    list_data.append(prob)

        
            plt.plot( list_angle, list_data, line_opt[gateno], label = f'{gate}' )
            plt.xlabel( f'Angle', fontsize = 12 )
        if calculate == 'rate':
            plt.plot( list_angle, [0.60205999132]*len(list_angle), 'ko', label = 'chiribella disrimination rate' )
            plt.ylabel('Discrimination rate', fontsize=12)
        elif calculate == 'prob':
            chiribella_error_prob = (3/(2*2**4))*(1 - np.sqrt(1 - 3**(-2)))
            plt.plot( list_angle, [chiribella_error_prob]*len(list_angle), 'ko', label = 'chiribella error' )
            plt.ylabel( 'Error probability', fontsize = 12 )
        
        plt.legend()
        plt.savefig( f'plot/hypothesis_distinguishing_{calculate}.pdf' )
        plt.savefig( f'plot/hypothesis_distinguishing_{calculate}.png' )
        plt.show()