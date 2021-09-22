import numpy as np


def newton_step(p_newton_old, dt, n):
    """
    Performs a Newtons step

    :param p_newton_old: Old pressure array
    :param dt: Time Step
    :param n: Grid Length
    :return: Newton stepped Pressure
    """
    eps = 1
    pold = p_newton_old
    pnew = p_newton_old
    omega = 0.1
    step = 0
    while eps > 1e-3 and step < 1000:
        pold = pnew
        gradient = np.gradient(pold, 1.0/(n))
        # Eliminate Zeros and Nans
        new_divide = np.zeros(n+1)
        for i in range(len(gradient)-1):
            if gradient[i] != 0:
                new_divide[i] = pold[i]/gradient[i]
            else:
                new_divide[i] = 0

        pnew = pold + omega * new_divide
        eps = np.linalg.norm(pnew-p_newton_old)
        print('At Step {}, eps: {}'.format(step, eps))
        step += 1

    return pnew


def calculate_pd(p_newton_new, p_newton_old, dt,
                 delta_y, n, alpha_d, lambda_d):
    """
    Calculate next pd time step
    :param p_newton_new: Pressure from Newton step
    :param p_newton_old: Old pressure values
    :param dt: Time step
    :param delta_y: Space step
    :param n: Total grid points
    :param alpha_d: Alpha value
    :param lambda_d: Lambda value
    :return:
    """
    pd = p_newton_old

    # Calculate Pd(N-1)
    rhs = 0
    dp01 = (p_newton_new[1] - p_newton_new[0]) / delta_y
    if dp01 != 0:
        term1 = (((-2 * p_newton_new[n - 1]) + p_newton_new[n - 2]) / (
                delta_y ** 2)) * ((1 + lambda_d) ** 2) * (
                        1.0 / (dp01 ** 2))
        term2 = (n - 1) * delta_y * ((1 + lambda_d) ** 2) * (
                (p_newton_new[n - 1] - (2 * p_newton_new[n - 1])) / (
                delta_y ** 2)) * (1.0 / (dp01 ** 2))
        term3 = alpha_d * (lambda_d ** 2) * (1 + (delta_y * (n - 1)))
        rhs = term1 + term2 - term3
        pd[n - 1] = p_newton_old[n - 1] + (dt * rhs)

    # Calculate Pd -> 1 to N-2
    for i in range(len(p_newton_new) - 2, 1, -1):#, 1, -1):
        rhs = 0

        # Old Method without equation Modification
        # Leave Now, scheme is unstable

        term1 = (p_newton_new[i + 1] - (2 * p_newton_new[i]) + p_newton_new[
            i - 1]) / (delta_y ** 2)
        term21 = alpha_d * p_newton_new[n - 1] * (i - 1)
        term22 = (((1 + lambda_d) ** 2) / lambda_d ** 2) * i * ((p_newton_new[
                                                                     n - 2] - (
                                                                         2 *
                                                                         p_newton_new[
                                                                             n - 1])) / delta_y ** 2) * \
                 p_newton_new[n - 1]
        term2 = ((p_newton_new[i + 1] - p_newton_new[i]) / delta_y) * (
                term21 - term22)

        # New Method with Modifications

        #term1 = (pd[i + 1] - (2 * pd[i]) + pd[i - 1]) / (delta_y ** 2)
        #term21 = alpha_d * pd[n - 1] * (i - 1)
        #term22 = (((1 + lambda_d) ** 2) / lambda_d ** 2) * i * ((pd[n - 2] - (2 * pd[n - 1])) / delta_y ** 2) * pd[n - 1]
        #term2 = ((pd[i + 1] - pd[i]) / delta_y) * (term21 - term22)

        rhs = ((1 + lambda_d) ** 2) * (term1 + term2)

        pd[i] += (dt * rhs)

        # Leave the next line for now. Some instabilities there
        #pd[i] = p_newton_old[i] + (dt * rhs)


    # Calculate Pd 0
    pd[0] = pd[1] + ((1 + lambda_d) / lambda_d) * pd[n - 1]

    return pd