# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:35:35 2019

@author: U80824876
"""

import numpy as np
import scipy.special as spec


def intdei(G, a, rho, n, h, nu, args=()):
    """

    intdei(G, a, n, args)

    This subroutine computes the integral of the following function
        G(kro, args) * J0(kro * rho) * kro,   for nu = 0
        G(kro, args) * J1(kro * rho) * kro^2, for nu = 1
    over the inverval kro = (a, infty)
    using double exponential-type quadrature formulas from
    [1] Golubovic et al. "Fast Computation of Sommerfeld Integral Tails via Direct
    Integration Based on Double Exponential-Type Quadrature Formulas,"
    IEEE Transactions on Antennas and Propagation, vol. 59, no. 2, February 2011

    Parameters
    ----------

    G : function
        A Python function or method used to integrate. If `G` takes many
        arguments, it is integrated along the axis corresponding to the
        first argument.
    a : double
        Lower limit of integration
    rho : double
        Spatial variable
    n : integer
        Number of integration points in the quadrature rule
    h : double
        Step size parameter, default h = 1.0/32.0
    nu : integer
        Order of transformation nu = 0, 1
    args : tuple, optional
        Extra arguments to pass to `G`

    Returns
    -------
    result : float
        Numerical value of the integral
            G(kro, args) * J_nu(kro * rho) * kro^(nu+1)
        over the semi-infinite interval kro = (a, infty)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    12NOV19     Ivica Stevanovic, OFCOM         First implementation in python

    """

    if not isinstance(args, tuple):
        args = (args,)

    def phi(t, h, b):
        """
        Transformation function specifically tailored for Sommerfeld Integral tails
        as given by equation (6) in [1]

        """
        arg = np.pi / 2.0 * np.sinh(t)

        result = np.pi / h * t * np.tanh(arg) + b * sech(arg)

        return result

    def dphi(t, h, b):
        """
        Derivative of the transformation function phi(t,h,b) w.r.t. t
        """
        arg = np.pi / 2.0 * np.sinh(t)

        result = np.pi / h * np.tanh(arg) + np.pi / h * t * np.power(
            sech(arg), 2.0
        ) * np.pi / 2.0 * np.cosh(t)

        result = result - b * np.tanh(arg) * sech(arg) * np.pi / 2.0 * np.cosh(t)

        return result

    def sech(arg):
        overflow_limit = 200
        x = min(arg, overflow_limit)

        return 1.0 / np.cosh(x)

    b = a * rho

    # Compute the abscissae as zeros of zero order Bessel function
    if nu == 0:

        x = np.array(spec.jn_zeros(0, n))

        y = 0.0 + 1j * 0.0

        for kk in range(0, len(x)):

            t = phi(h * x[kk] / np.pi, h, b)  # (6)

            # Compute the corresponding weights

            w = 2.0 / (np.pi * x[kk] * np.power(spec.jv(1, x[kk]), 2.0))  # (13)

            f0 = t / (rho * rho) * G(t / rho, *args) * spec.jv(0, t)  # (8)

            t0 = w * f0 * dphi(h * x[kk] / np.pi, h, b)  # (14)

            y = y + t0

    else:

        x = np.array(spec.jn_zeros(1, n))

        y = 0.0 + 1j * 0.0

        for kk in range(0, len(x)):

            t = phi(h * x[kk] / np.pi, h, b)  # (6)

            # Compute the corresponding weights

            w = 2.0 / (np.pi * x[kk] * np.power(spec.jv(2, x[kk]), 2.0))  # (13)

            f1 = t * t / (np.power(rho, 3.0)) * G(t / rho, *args) * spec.jv(1, t)  # (8)

            t1 = w * f1 * dphi(h * x[kk] / np.pi, h, b)  # (14)

            y = y + t1

        y = y + (2.0 - 0.5 * b * h) * (
            b * b / (np.power(rho, 3.0)) * G(b / rho, *args) * spec.jv(1, b)
        )

    return h * y
