from matplotlib import pyplot as plt
import pandas as pd


def plot_tuning_logs(path="hyper_tuning/tuning_log.txt"):

    full_text = open(path, "r").read()
    trial_list = full_text.split("Search: Running Trial")
    all_hyperpars = []
    all_val_losses = []
    for trial in trial_list[1:]:
        try:
            hyperparameters = trial.split("HYPERPARAMETERS:")[1].split('\n')[0].split(';')
            hyperparameters[1] = f"{float(hyperparameters[1]):.0e}"  # display learning rate as e
            val_loss = trial.split("\nval_loss: ")[1].split('\n')[0]
            all_hyperpars.append(hyperparameters)
            all_val_losses.append(float(val_loss))
        except:
            pass

    # get strings from hyperparameters
    all_labels = []
    for hyperpar in all_hyperpars:
        label = ""
        for i in range(len(hyperpar)):
            label += hyperpar[i] + " "
        all_labels.append(label)

    print(all_labels)

    # sort by val_loss
    all_val_losses, all_labels = zip(*sorted(zip(all_val_losses, all_labels), reverse=True))

    # Plot the figure.
    freq_series = pd.Series(all_val_losses)
    plt.figure(figsize=(12, 12))
    ax = freq_series.plot(kind="barh")
    ax.set_xscale('log')
    ax.set_title("Hyperparameter Tuning")
    ax.set_xlabel("Validation Loss")
    ax.set_yticklabels(all_labels)

    plt.show()


def plot_federated_logs(path="logs/bearing_experiment-2/1.txt"):


    lstm_rounds = []
    fft_rounds = []
    lstm_val_losses = []


    full_text = open(path, "r").read()

    all_rounds = full_text.split("Round ")[1:]
    lstm_rounds = [round.split("\n")[1] for round in all_rounds]
    fft_rounds = [round.split("\n")[2] for round in all_rounds]

    lstm_val_losses = [lstm_round.split('val_loss: ')[1].split(' -')[0] for lstm_round in lstm_rounds]
    fft_val_losses = [fft_round.split('val_loss: ')[1].split(' -')[0] for fft_round in fft_rounds]

    lstm_val_losses = [float(loss) for loss in lstm_val_losses]
    fft_val_losses = [float(loss) for loss in fft_val_losses]

    # Plot the figure.
    plt.plot(lstm_val_losses, label="LSTM")
    plt.plot(fft_val_losses, label="FFT")
    plt.title("Validation Loss")
    plt.xlabel("Round")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()







if __name__ == "__main__":
    plot_federated_logs()