import numpy as np
import PIL 
import matplotlib as mpl
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

results = pd.read_csv('track_of_state.csv')

results.head()

unique_values_A = results['episode_num'].unique()
num_episodes = len(unique_values_A)

# Create a colormap with a unique color for each episode
color_map = plt.cm.get_cmap('tab10', num_episodes)
color_cycle = cycle(color_map(range(num_episodes)))

plt.figure()

for episode in unique_values_A:
    episode_data = results[results['episode_num'] == episode]
    
    # Extract x-values (from column B) and y-values (from column C)
    x = episode_data['state_0']
    y = episode_data['state_2']
    
    color = next(color_cycle)
    
    # Create a plot for each episode
    # plt.plot(x, y, color=color)
    plt.scatter(x, y, color=color, marker='o', s=50, zorder=2)

    
plt.title(f'Episodes')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.grid()
plt.show()
