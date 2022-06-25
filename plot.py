import matplotlib.pyplot as plt
import numpy as np


labels = ['mat128_32', 'mat128_64', 'mat128_128', 'mat128_256','mat512_32', 'mat512_64', 'mat512_128', 'mat512_256']
prog2 = [0.0008524, 0.0184793, 0.2873067, 2.6378241, 0.0054755, 0.1051129, 1.2264283, 10.8601548]#prog2
prog1 = [0.0006655, 0.0044641, 0.0436721, 0.2530478, 0.0025034, 0.0179528, 0.1200312, 0.6376904] #prog1


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, prog2, width, label='Abordagem por linhas')
rects2 = ax.bar(x + width/2, prog1, width, label='Abordagem por colunas')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Tempo (s)')
ax.set_title('Ficheiros')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects2, padding=3)
ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()