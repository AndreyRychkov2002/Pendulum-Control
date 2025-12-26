import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi
import matplotlib.animation as animation
from scipy.linalg import solve_continuous_are

# --- Physical Constants ---
G = 9.81
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
L = L1 + L2
dt = 0.01
t_stop = 10
t = np.arange(0, t_stop, dt)

# --- Disturbance Settings ---
NOISE_MAGNITUDE = 0.5  # Very low for underactuated stability testing
KICK_STRENGTH = 0.0


def get_lqr_gain(inverted=False):
    # Mass Matrix M at equilibrium (cos(delta) = 1)
    M = np.array([
        [(M1 + M2) * L1 ** 2, M2 * L1 * L2],
        [M2 * L1 * L2, M2 * L2 ** 2]
    ])

    # Gravity/Stiffness Matrix Ks
    coeff = -1 if inverted else 1
    Ks = coeff * np.array([
        [(M1 + M2) * G * L1, 0],
        [0, M2 * G * L2]
    ])

    Minv = np.linalg.inv(M)

    # State Space A
    A = np.zeros((4, 4))
    A[0, 1], A[2, 3] = 1, 1
    A21 = -Minv @ Ks
    A[1, 0], A[1, 2] = A21[0, 0], A21[0, 1]
    A[3, 0], A[3, 2] = A21[1, 0], A21[1, 1]

    # UNDERACTUATED B: Only Joint 1 is driven.
    # Torque is applied to q1 directly.
    B_q = np.array([[1], [0]])
    B = np.zeros((4, 1))
    B[[1, 3], :] = Minv @ B_q

    # Tuning: High penalty on angles, low on velocities
    Q = np.diag([1000, 100, 1000, 100])
    R = np.array([[0.1]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


K_stable = get_lqr_gain(inverted=False)
K_unstable = get_lqr_gain(inverted=True)


def derivs(state, u_val=0, d=None):
    u = np.array([u_val, 0.0])  # Only joint 1 gets the scalar u_val
    d = d if d is not None else np.zeros(2)
    torque = u + d

    th1, w1, th2, w2 = state
    delta = th2 - th1

    # Full Non-linear Mass Matrix
    m11 = (M1 + M2) * L1 ** 2
    m12 = M2 * L1 * L2 * cos(delta)
    m21 = M2 * L1 * L2 * cos(delta)
    m22 = M2 * L2 ** 2
    MM = np.array([[m11, m12], [m21, m22]])

    # Coriolis and Gravity
    c1 = M2 * L1 * L2 * (w2 ** 2) * sin(delta) + (M1 + M2) * G * L1 * sin(th1)
    c2 = -M2 * L1 * L2 * (w1 ** 2) * sin(delta) + M2 * G * L2 * sin(th2)
    C = np.array([c1, c2])

    accels = np.linalg.solve(MM, torque - C)

    return np.array([w1, accels[0], w2, accels[1]])


# --- Simulation Execution ---
# Underactuated systems need smaller initial perturbations to stay in linear range
state_init = np.radians([40.0, 0.0, -10.0, 0.0])

y_un = np.zeros((len(t), 4));
y_st = np.zeros((len(t), 4));
y_up = np.zeros((len(t), 4))
# Set starting positions (Unstable needs to start near PI)
y_un[0] = state_init
y_st[0] = state_init
y_up[0] = state_init + np.array([pi, 0, pi, 0])

noise = np.random.normal(0, NOISE_MAGNITUDE, (len(t), 2))

for i in range(1, len(t)):
    d = noise[i - 1]
    # Uncontrolled
    y_un[i] = y_un[i - 1] + derivs(y_un[i - 1], d=d) * dt
    # Stable
    u_st = (-K_stable @ y_st[i - 1])[0]
    y_st[i] = y_st[i - 1] + derivs(y_st[i - 1], u_st, d) * dt
    # Unstable (PI Target)
    err_up = y_up[i - 1] - np.array([pi, 0, pi, 0])
    err_up[0] = (err_up[0] + pi) % (2 * pi) - pi
    err_up[2] = (err_up[2] + pi) % (2 * pi) - pi
    u_up = (-K_unstable @ err_up)[0]
    y_up[i] = y_up[i - 1] + derivs(y_up[i - 1], u_up, d) * dt

# --- Plotting ---
fig, axs = plt.subplots(4, 2, figsize=(12, 10))
fig.suptitle("Underactuated Control (Torque only on Joint 1)", fontsize=16)
titles = ["Stable Link 1", "Unstable Link 1", "Stable Link 2", "Unstable Link 2"]
data_labels = [r"$\theta_1$", r"$\dot{\theta}_1$", r"$\theta_2$", r"$\dot{\theta}_2$"]

for i in range(4):
    axs[i, 0].plot(t, y_st[:, i], 'b')
    axs[i, 1].plot(t, y_up[:, i], 'g')
    axs[i, 0].set_ylabel(data_labels[i])
    axs[i, 0].grid(True);
    axs[i, 1].grid(True)

fig_anim, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
lines = []
for ax, title in zip([ax1, ax2, ax3], ["Uncontrolled", "Underactuated Stable", "Underactuated Unstable"]):
    ax.set_xlim(-L, L);
    ax.set_ylim(-L, L);
    ax.set_aspect('equal')
    ax.grid();
    ax.set_title(title)
    line, = ax.plot([], [], 'o-', lw=2)
    lines.append(line)


def animate(i):
    for idx, data in enumerate([y_un, y_st, y_up]):
        x1 = L1 * sin(data[i, 0])
        y1 = -L1 * cos(data[i, 0])
        x2 = L2 * sin(data[i, 2]) + x1
        y2 = -L2 * cos(data[i, 2]) + y1
        lines[idx].set_data([0, x1, x2], [0, y1, y2])
    return *lines,


ani = animation.FuncAnimation(fig_anim, animate, frames=len(t), interval=dt*1000, blit=True)
writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=512)
ani.save('pendulum.gif', writer=writer)
plt.tight_layout()
plt.show()