import numpy as np
import matplotlib.pyplot as plt


# Parameters Setting
# Variables Note: sigma = tilde{L}/tilde{mu}, s = alpha_hat = alpha * tilde{mu}
sigma_list = [1.5, 2, 3]
gamma_list = [0.25, 0.5, 0.75, 1, 1.1, 1.2, 1.5, 2]
print_flag = True


# Utility Functions
def s_domain(sigma, gamma):
    '''Calculate the domain of s = alpha * tilde{mu}'''

    eps = 1e-5
    if gamma < 1:
        return np.linspace(0, 1/sigma, num=200)
    else:
        return np.linspace(1 - 1/gamma + eps, 1/sigma, num=200)


def plot_one_gamma(ax, s, arrays, gamma, c_idx=0):
    '''Plot arrays wrt s for a given gamma'''

    gamma_show = np.around(gamma, decimals=2)
    ax.plot(s, arrays, label=r"$\gamma = $" + str(gamma_show), color='C'+str(c_idx))


##### Main Script
path_base = "../figures/variables_dependency/"
gamma_list.reverse()  # reversed order
gamma_end_array = np.array(sigma_list) / (np.array(sigma_list) - 1)

### Convergence Rate and Convergence Radius
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for g_idx, gamma in enumerate(gamma_list):
    # Determine domain s (depending on gamma and sigma)
    s = s_domain(1, gamma)
    # Convergence Rate: sqrt(gamma) * sqrt(1 - s)
    conv_rate = np.sqrt(gamma) * np.sqrt(1 - s)
    # Convergence Radius: R* = sqrt(s)/(1 - sqrt(gamma*(1-s)))
    conv_radius = np.sqrt(s) / (1 - np.sqrt(gamma) * np.sqrt(1 - s))
    # Plot
    color_idx = len(gamma_list) - g_idx - 1
    plot_one_gamma(ax1, s, conv_rate, gamma, color_idx)
    plot_one_gamma(ax2, s, conv_radius, gamma, color_idx)

# Show plot (Convergence Rate)
ax1.set_title(r"Convergence Rate")
ax1.set_ylabel(r"$rate(\gamma, \tilde{\alpha})$")
ax1.set_xlabel(r"(Scaled) Constant Step-size $\tilde{\alpha}$")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.0])
ax1.legend()
ax1.grid()
path_name = path_base + "convergence_rate.jpg"
if print_flag: fig1.savefig(path_name, dpi=300)
#fig1.show()

# Show plot (Convergence Radius)
ax2.set_title(r"(Normalized) Convergence Radius")
ax2.set_ylabel(r"$R^*_{normalized} (\gamma, \tilde{\alpha})$")
ax2.set_xlabel(r"(Scaled) Constant Step-size $\tilde{\alpha}$")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 10])
ax2.legend()
ax2.grid()
path_name = path_base + "convergence_radius.jpg"
if print_flag: fig2.savefig(path_name, dpi=300)
#fig2.show()


### Consensus Diameter
diameter_plots = [plt.subplots() for _ in range(len(sigma_list))]
for s_idx, sigma in enumerate(sigma_list):
    gamma_list_2 = [gamma for gamma in gamma_list if gamma < gamma_end_array[s_idx]]
    for g_idx, gamma in enumerate(gamma_list_2):
        # Determine domain s (depending on gamma and sigma)
        s = s_domain(sigma, gamma)
        # Convergence Radius: R* = sqrt(s)/(1 - sqrt(gamma*(1-s)))
        conv_radius = np.sqrt(s) / (1 - np.sqrt(gamma) * np.sqrt(1 - s))
        # Consensus Diameter: D* = delta*s*(1 + R* * sqrt(gamma*sigma))
        cons_diam = sigma * s * (1 + np.sqrt(gamma * sigma) * conv_radius)
        # Plot
        color_idx = len(gamma_list_2) - g_idx - 1
        plot_one_gamma(diameter_plots[s_idx][1], s, cons_diam, gamma, color_idx)
        
    # Show plot (Consensus Diameter)
    diameter_plots[s_idx][1].set_title(r"(Normalized) Consensus Diameter for $\sigma = $" + str(sigma))
    diameter_plots[s_idx][1].set_ylabel(r"$D^*_{normalized} (\sigma, \gamma, \tilde{\alpha})$")
    diameter_plots[s_idx][1].set_xlabel(r"(Scaled) Constant Step-size $\tilde{\alpha}$")
    diameter_plots[s_idx][1].set_xlim([0, 2/3])
    diameter_plots[s_idx][1].set_ylim([0, 20])
    diameter_plots[s_idx][1].legend(loc=1)
    diameter_plots[s_idx][1].grid()
    path_name = path_base + "consensus_diam_" + str(s_idx+1) + ".jpg"
    if print_flag: diameter_plots[s_idx][0].savefig(path_name, dpi=300)
    #diameter_plots[s_idx][0].show()