from matplotlib import pyplot as plt
import pandas as pd

# read from log file
path = "logs/tuning_log.txt"
full_text = open(path, "r").read()
trial_list = full_text.split("Search: Running Trial")
trial_results = []
all_hyperpars = []
all_val_losses = []
for trial in trial_list[1:]:
    try:
        hyperparameters = trial.split("HYPERPARAMETERS:")[1].split('\n')[0].split(';')
        val_loss = trial.split("\nval_loss: ")[1].split('\n')[0]
        all_hyperpars.append(hyperparameters)
        all_val_losses.append(float(val_loss))
        print(hyperparameters, val_loss)
    except:
        pass

# sort all_hyperpars alphabetically

print(all_hyperpars)
print(all_val_losses)


# Plot the figure.
freq_series = pd.Series(all_val_losses)
plt.figure(figsize=(12, 12))
ax = freq_series.plot(kind="barh")
ax.set_xscale('log')
ax.set_title("Hyperparameter Tuning")
ax.set_xlabel("Validation Loss")
ax.set_yticklabels(all_hyperpars)

plt.show()
