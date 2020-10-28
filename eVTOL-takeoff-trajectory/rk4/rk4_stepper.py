import numpy as np


def rk4_stepper(dt, t, x, f, u=None):
    if u is None:
        k1 = dt * f(t, x)
        k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * f(t + dt, x + k3)
    else:
        k1 = dt * f(t, x, u)
        k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, u)
        k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2, u)
        k4 = dt * f(t + dt, x + k3, u)

    x_new = x + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    return x_new


def cannonball_ode(t, x):
    ax = 0
    ay = -9.8
    vx = x[2]
    vy = x[3]
    dxdt = np.array([vx, vy, ax, ay])
    return dxdt


if __name__ == '__main__':
    tf = 2*10*np.sqrt(2)/9.8
    num_steps = 500
    dt = tf / num_steps
    t = 0
    x = np.zeros((4, num_steps + 1))

    x[:, 0] = np.array([0, 0, 10*np.sqrt(2), 10*np.sqrt(2)])
    for i in range(num_steps):
        x[:, i + 1] = rk4_stepper(dt, t, x[:, i], cannonball_ode)
        t += dt
    print('final value of x position = {}'.format(x[0, -1]))
    print('range of cannonball, vx0 * t = {}'.format(10*np.sqrt(2)*tf))
    print('rel_error = {}'.format(1 - x[0, -1]/(10*np.sqrt(2)*tf)))
    show_plot = False
    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(x[0, :], x[1, :])
        plt.show()
