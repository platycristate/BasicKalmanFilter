import numpy as np
from filters import KalmanFilter
import matplotlib.pyplot as plt
plt.style.use('dark_background')


Δt = 0.01 
N = 1000
measurement_var = 0.1
command_var = 0.2
mean_acc = 0
mean_init = np.array([0, 0.95]).reshape(2, 1)
cov_init = np.array([[0.86, 0], [0, 4]])
true_position = 0
true_velocity = 1

def generate_command(acc):
    return acc + np.random.normal(
        loc=mean_acc, scale=command_var, size=(1, 1))

if __name__ == '__main__':
    kf = KalmanFilter(mean_init, cov_init, command_var, measurement_var, Δt)
    means, covs = [], []
    measurement = mean_init[0] + np.random.rand() * np.sqrt(measurement_var)

    pred_positions = []
    true_positions = []
    pred_velocities, true_velocities = [], []
    pred_covs, pred_covs_v = [], []
    accelarations = [np.random.normal(0.0, 4) for i in range(N)]
    for i, acc in zip(range(N), accelarations):
        command = generate_command(acc)
        true_position += Δt * true_velocity
        true_velocity += Δt * acc
        if i % 20 == 0:
            measurement = true_position + np.random.rand() * np.sqrt(measurement_var)
            kf.update(command, measurement)
        else:
            kf.update(command)

        pred_positions.append(kf.mean[0])
        pred_velocities.append(kf.mean[1])
        pred_covs.append(kf.cov[1, 1])
        pred_covs_v.append(kf.cov[0, 0])
        true_positions.append(true_position)
        true_velocities.append(true_velocity)
    
    # Plot results
    fig, axs = plt.subplots(2, 1, dpi=120)
    axs[0].plot(true_positions, label='True')
    axs[0].plot(pred_positions, label='Pred')
    axs[0].plot([mean + 2 * np.sqrt(cov) for mean, cov in zip(true_positions, pred_covs)], '--')
    axs[0].plot([mean - 2 * np.sqrt(cov) for mean, cov in zip(true_positions, pred_covs)], '--')
    axs[0].set(xlabel='Time', ylabel='Position')
    axs[0].legend(title='Position')

    axs[1].plot(true_velocities, label='True')
    axs[1].plot(pred_velocities, label='Pred')
    axs[1].plot([mean + 2 * np.sqrt(cov) for mean, cov in zip(true_velocities, pred_covs_v)], '--')
    axs[1].plot([mean - 2 * np.sqrt(cov) for mean, cov in zip(true_velocities, pred_covs_v)], '--')
    axs[1].set(xlabel='Time', ylabel='Velocity')
    axs[1].legend(title='Velocity')
    plt.show()



