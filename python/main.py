from cProfile import label
from multiprocessing.spawn import import_main_path
import numpy as np
from matplotlib import pyplot as plt

from kf import KF


initial_x = 5
initial_v = 10
accel_variance = 1
meas_variance = 0.3**2

real_x = 0
real_v = 5

DT = 0.1
TIME_STEPS = 200
UPDATE_STEP = 5
real_xs = []
real_vs = 5*np.sin(np.linspace(0,TIME_STEPS//20,TIME_STEPS))


kf = KF(initial_x, initial_v, accel_variance)
kf.track_history()

for step in range(TIME_STEPS):

    real_x += real_vs[step] * DT

    real_xs.append(real_x)
    #real_vs.append(real_v)

    kf.predict(DT)
    if step % UPDATE_STEP == 0 and step != 0:
        kf.update(real_x + np.random.randn() * np.sqrt(meas_variance), meas_variance)

x_history, p_history = kf.get_history()

fig, axs = plt.subplots(2, 1)

axs[0].set_title("Position")
axs[0].plot([x[0] for x in x_history], "r", label="State estimate")
axs[0].plot(real_xs, 'g', label = "Real Position")
axs[0].plot([x[0] + 2*np.sqrt(p[0][0]) for x,p in zip(x_history,p_history)], "r--", label="Uncertainty")
axs[0].plot([x[0] - 2*np.sqrt(p[0][0]) for x,p in zip(x_history,p_history)], "r--")

axs[1].set_title("Velocity")
axs[1].plot([x[1] for x in x_history], "r",  label="State estimate")
axs[1].plot(real_vs, 'g', label = "Real Position")
axs[1].plot([x[1] + 2*np.sqrt(p[1][1]) for x,p in zip(x_history,p_history)], "r--", label="Uncertainty")
axs[1].plot([x[1] - 2*np.sqrt(p[1][1]) for x,p in zip(x_history,p_history)], "r--")

plt.legend()
plt.show()
