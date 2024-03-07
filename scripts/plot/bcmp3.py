import matplotlib.pyplot as plt
import pickle
import sinter
from typing import List
from utils import calculate_ler

samples: List[sinter.TaskStats] = pickle.load(open('sp_data/bcmp3.pkl', 'rb'))

plt.figure(figsize=(8, 6), dpi=100)
ax = plt.subplot(111)

legend = {
        0: 'Bandage',
        1: 'Tradition',
    }
sinter.plot_error_rate(
    ax=ax,
    stats = samples,
    group_func=lambda stat: f'{legend[stat.json_metadata["device_index"]]}|{stat.json_metadata["initial_state"]}>',
    x_func=lambda stat: stat.json_metadata["specified_holding_cycle"],
    highlight_max_likelihood_factor=1,
    plot_args_func=lambda index, curve_id: {
        'linestyle': '--' if index in [2, 3] else '-',
        'color': 'C0' if index in [0, 2] else 'C2',
        'marker': 'o' if index in [0, 2] else '^',
    },
)
ax.grid()
ax.set_title("LER vs Global Shell Size")
ax.set_ylabel("LER")
ax.set_xlabel("Global Shell Size")
ax.legend()
plt.savefig('bcmp3.pdf', format='pdf')
plt.show()

result = {"0": {"0": 0, "+": 0}, "1": {"0": 0, "+": 0}}
for i in range(2):
    for s in ["0", "+"]:
        filter_func = (
            lambda stat: stat.json_metadata["initial_state"] == s
            and stat.json_metadata["device_index"] == i
        )
        print(
            f"Lowest LER for curve {i}|{s}>: {min([calculate_ler(stat) for stat in filter(filter_func, samples)])}"
        )
        result[str(i)][s] = min(
            [calculate_ler(stat) for stat in filter(filter_func, samples)]
        )

print('0 state improvement: ', (result["1"]["0"] - result["0"]["0"]) / result["1"]["0"])
print('+ state improvement: ', (result["1"]["+"] - result["0"]["+"]) / result["1"]["+"])