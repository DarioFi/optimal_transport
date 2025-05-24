import matplotlib.pyplot as plt
import numpy as np
from opt_trans.data_handling.experiment_data import Database

folder = '../run_logs/baseline_runtime_performance_logarithm/'
db = Database.populate_from_folder(folder)
data = db.index_on('instance_arguments')

print(data.keys())

# plot sqrt(x) and log(x + 1)
x = np.linspace(0, 1, 100)
y1 = np.sqrt(x)
plt.plot(x, y1, label='sqrt(x)', color='blue')
alpha = 2

for alpha in [10, 20, 40, 100]:
    y2 = np.log(alpha * x + 1) / np.log(alpha + 1)
    plt.plot(x, y2, label=rf'$\log({alpha} x + 1)$')
plt.legend()
plt.show()
