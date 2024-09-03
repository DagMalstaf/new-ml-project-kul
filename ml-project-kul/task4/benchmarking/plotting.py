import matplotlib.pyplot as plt
import numpy as np

agents = ['MCTS Agent', 'AlphaZero Agent', 'GNN Agent']
simulations = ['10 MCTS SIMS', '25 MCTS SIMS', '100 MCTS SIMS']

mcts_data = [86.5, 89.5, 94.5]
alphazero_data = [47.5, 54.0, 61.5]
gnn_data = [98.5, 94.5, 96.0]

x = np.arange(len(simulations))  
width = 0.2 

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#4e79a7', '#f28e2b', '#76b7b2']

rects1 = ax.bar(x - width, mcts_data, width, label='MCTS Agent', color=colors[0])
rects2 = ax.bar(x, alphazero_data, width, label='AlphaZero Agent', color=colors[1])
rects3 = ax.bar(x + width, gnn_data, width, label='GNN Agent', color=colors[2])

ax.set_ylabel('Win Rate against Random (%)', fontsize=14)
ax.set_ylim(0, 100)
ax.set_title('Performance of Agents by Number of MCTS Simulations (Rounds=100)', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(simulations, fontsize=12)

legend = ax.legend(title='Agents', title_fontsize=12, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
