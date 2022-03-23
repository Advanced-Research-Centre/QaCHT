from main_circuit import *
import matplotlib.pyplot as plt

def distinguishing_probability(theta):
    list_opts = [ 'swap', 'identity' ]
    dict_swap = pickle.load(open(f'data/{list_opts[0]}_dict_prob_{theta}.p', "rb"))
    dict_identity = pickle.load(open(f'data/{list_opts[1]}_dict_prob_{theta}.p', "rb"))

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
            if k not in list_key_identity:
                x += dict_swap[k]
            else:
                x += dict_identity[k]
    return x

if __name__ == "__main__":

    list_prob = []
    list_angle = []
    for theta in np.arange(0, 2*np.pi, 0.02):
        prob = distinguishing_probability(theta)
        list_prob.append(prob)
        list_angle.append(theta)
    
    plt.plot( list_angle, list_prob, 'k-' )
    plt.xlabel( 'Angle in RY', fontsize = 12 )
    plt.ylabel( 'Hypothesis distinguishing probability', fontsize = 12 )
    plt.savefig( 'plot/hypothesis_distinguishing_probability.pdf' )
    plt.savefig( 'plot/hypothesis_distinguishing_probability.png' )
    plt.show()