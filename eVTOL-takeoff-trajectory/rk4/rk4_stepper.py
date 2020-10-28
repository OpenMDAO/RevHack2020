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


def example_func(t, x):
    return 0.5 * (t - x)


if __name__ == '__main__':
    tf = 2
    num_steps = 10
    dt = tf / num_steps
    t = 0
    x = np.zeros(num_steps + 1)

    x[0] = 1
    for i in range(num_steps):
        x[i + 1] = rk4_stepper(dt, t, x[i], example_func)
        t += dt
    print('Value of x at t = 2 is {}'.format(x[-1]))
    print('Expected value is 1.103639')
