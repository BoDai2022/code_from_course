import math
import numpy as np
import matplotlib.pyplot as plt
# The plot shows how the normalized essential bit content changes as the confidence parameter 𝛿 changes, for different numbers of coin flips N. 
# thus how much data is needed to reach a certain confidence level in probabilistic scenarios.
def m_choose_n(m,n):
    return int(math.factorial(m)/(math.factorial(m-n) * math.factorial(n)))
theta = 0.8
def calc_normalised_essential_bit_content(N,delta):
    len_smallest_set = 0
    n_head = N
    prob_acc = 0
    while(n_head >=0):
        prob_has_n_head_seq = theta**n_head * (1-theta)**(N-n_head)
        n_seq = m_choose_n(N, n_head)
        if(prob_acc + prob_has_n_head_seq * n_seq >= 1-delta):
            n_needed = np.ceil((1 - delta -prob_acc)/prob_has_n_head_seq)
            prob_acc += n_needed * prob_has_n_head_seq
            len_smallest_set += n_needed
            return (1/N) * math.log2(len_smallest_set)
        else:
            prob_acc = prob_acc + prob_has_n_head_seq * n_seq
            len_smallest_set += n_seq
        n_head -= 1
    return 1  # all subsets needed
n_flips = [10,100,500,800,1002]
delta_list = np.linspace(0,1,100)[:-1]  # exclude 1
data = []
for N in n_flips:
    normlised_essential_bit_content_list = []
    for delta in delta_list:
        normlised_essential_bit_content_list.append(calc_normalised_essential_bit_content(N,delta))
    data.append(normlised_essential_bit_content_list)
# Plot the data
for i, N in enumerate(n_flips):
    plt.plot(delta_list, data[i], label=f'N={N}')
# Add labels and legend
plt.xlabel('Delta')
plt.ylabel('Normalized Essential Bit Content')
plt.title('Normalized Essential Bit Content vs. Delta')
plt.legend()
# Show the plot
plt.grid()
# plt.show()
plt.savefig('normalized_essential_bit_content.png')

