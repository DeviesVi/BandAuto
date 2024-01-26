import matplotlib.pyplot as plt
import pickle
import sinter
from typing import List

samples: List[sinter.TaskStats] = pickle.load(open('sp_data/nshell_p_cor.pkl', 'rb'))

plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot(111)

sinter.plot_error_rate(
    ax=ax,
    stats = samples,
    group_func=lambda stat: f"p={stat.json_metadata['physical_errors']['u2']}|{stat.json_metadata['initial_state']}>",
    x_func=lambda stat: stat.json_metadata["specified_holding_cycle"],
    highlight_max_likelihood_factor=1,
    plot_args_func=lambda index, curve_id: {'marker': None},
    filter_func=lambda stat: stat.json_metadata['initial_state'] == '0',
)
# Set y to log
ax.set_yscale('log')

ax.grid()
ax.set_title("LER vs Global Shell Size")
ax.set_ylabel("LER")
ax.set_xlabel("Global Shell Size")
ax.legend()
plt.savefig('nshell_p_cor.pdf', format='pdf')
plt.show()