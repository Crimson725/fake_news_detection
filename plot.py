import matplotlib.pyplot as plt
from utils.helper_function import load_metrics
import config

train_loss_list, valid_loss_list, global_steps_list = load_metrics(config.DESTINATION_PATH + "/" + 'metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Validation')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()