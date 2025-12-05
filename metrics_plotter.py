import matplotlib.pyplot as plt
import numpy as np

class MetricsPlotter:
    def __init__(self):
        self.metrics = {
            "qloss": [],
            "piloss": [],
            "etaloss": [],
            "eta": [],
            "kl": [],
            "return": []
        }

    def plot_metrics(self, new_metrics, iteration):
        for key in self.metrics:
            if key in new_metrics:
                value = new_metrics[key]
                # Handle different data types
                if isinstance(value, (list, tuple)):
                    # Skip empty lists
                    if not value:
                        continue
                    # Flat list - extend with all values
                    self.metrics[key].extend(value)
                else:
                    # Scalar value
                    self.metrics[key].append(value)
        
        subplot_idx = 1
        for key, values in self.metrics.items():

            ax = plt.subplot(2, 3, subplot_idx)
            if len(values) > 0:
                ax.plot(values)
            ax.set_title(key)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(key)
            subplot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'archive/training_pictures/training_metrics_{iteration}.png')
        plt.close()