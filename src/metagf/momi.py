#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:09:09 2019


This library of functions implements DE tanh-sinh integration routines and general weighted averages
as defined in the following paper:

Krzysztof A. Michalski & Juan R. Mosig (2016) Efficient computation of
Sommerfeld integral tails – methods and algorithms, Journal of Electromagnetic Waves and
Applications, 30:3, 281-317
https://doi.org/10.1080/09205071.2015.1129915

@author: Ivica Stevanovic (OFCOM)
"""

import numpy as np

def partExtrap(G, rho, z, a, nu, eps, kmax, maxlev, args):
    ''' This function computes the integral of function G over a semi-infinite interval [a, infty)
        using a general partition-extrapolation algorithm via MosigMichalski
        according to Michalski&Mosig's paper in
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additional arguments to be passed to `G`
        a:      lower limit of the bounded integral
        nu:     order of Bessel function in the integrand
        eps:    square root of the desired precision
        kmax:   max order of the general WA algorithm
        maxlev: max level in DE tanh-sinh numerical integration
        
    '''
    

    if rho > 0:
        q = np.pi/rho
    else:
        q = np.pi/z
        
    alpha = 0.5-nu # check this
    
    mu = 2
    
    X = np.zeros(kmax+1, dtype='float64')
    R = np.zeros(kmax+1, dtype='complex128')
    
    s = 0
    
    old = 0
    
    for k in range(0, kmax+1):    
        if k == 0:
            X[k] = a + q
            u = tanhSinhQuad(G, args, a, X[0], eps, maxlev)
        else:
            X[k] = X[k-1] + q
            u = tanhSinhQuad(G, args, X[k-1], X[k], eps, maxlev)
        s = s + u
        
        o = omega(k, q, z, alpha, X)
        
        val = MosigMichalski(mu, k, s, o, X, R)
        
        # convergence check
        if (k > 1 and np.abs(val-old) < eps * np.abs(val)):
            break
        
        old = val
        
    return val


def MosigMichalski(mu, k, s, omega, X, R):
    ''' This function implements the weighted averages method via the
        Mosig-Michalski algorithm
        according to Michalski&Mosig's paper in
        https://doi.org/10.1080/09205071.2015.1129915
        
        mu:     convergence (1 - logarithmic, 2 - otherwise)
        k:      instantaneous transformation order (k = 0,...,kmax)
        s:      kth partial sum
        omega:  ratio of the current and previous remainder estimates at the transformation order k
        X:      an array of dimension k+1: xn = beta + n, beta usually set to 1
        R:      an array of dimension k+1: containing remainders 
    '''
    # The following values are recommended by Michalski&Mosig:
    
    R[k] = s
    
    for j in range(1, k+1):
        
        d = X[k-j+1] - X[k-j]
        
        #print "d = ", d
        #print "mu = ", mu
        #print "k, j, X[k-j] = ", k, j, X[k-j]
        
        eta = omega/( 1.0 + mu * (j-1.0) * (d+0.0)/(X[k-j]+0.0)  )
        
        R[k-j] = ( R[k-j+1] - eta * R[k-j] )/(1.0-eta)

    return R[0]

def omega(k, q, z, alpha, X):
    ''' This function computes the ratio of the current and previous 
        remainder estimates at the transformation order k in
        Mosig-Michalski algorithm 7
        according to Michalski&Mosig's paper in
        https://doi.org/10.1080/09205071.2015.1129915
        
        k:      instantaneous transformation order (k = 0,...,kmax)
        q:      equidistant spacing
        z:      source-observer distance in z-direction
        alpha:  0.5-nu, where nu is the order of Bessel function in the integrand
        X:      equidistant points (in rho)
    '''
    if k == 0:
        o = 0
    else:
        o = -np.exp(-q*z)*np.power( (X[k-1]/X[k]), alpha)
        
    return o

def tanhSinhQuad(G, args, a, b, eps, maxlev):
    ''' This function computes the integral of function G over a bounded interval [a, b]
        using an automatic tanh-sinh quadrature rule
        according to Michalski&Mosig's paper in
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additional parameters to be passed to `G`
        a:      lower limit of the bounded integral
        b:      upper limit of the bounded integral
        eps:    square root of the desired precision
        maxlev: order of the tanh-sinh integration 
        
        
        Function G(kro) should be programmed as a two-parameter procedure G(c, d), 
        where kro = c + d, c = a or b, and d = ±σ δ , so that
        the singular part may be computed using d directly.
        For example, if the integrand comprises a term (kro−a)^(-alpha) (b-kro)^(-beta)
        with alpha > 0 and beta > 0, this procedure should compute it as
        compute it as 
        d^(-alpha) * (b-kro)^(-beta) when d > 0
        and as
        (kro - a)^(-alpha) * (-d)^(-beta) when d < 0
        
    '''
    # The following values are recommended by Michalski&Mosig:
    
    eta = 1
    
    kappa = 1.0e-15
    
    nmax = 24
    
    h = 1.5
        
    val = 0.0+1j*0.0
    
    sigma = (b-a)/2.0
    
    gamma = (b+a)/2.0
    
    m = 0

    eh  = np.exp(h)

    c = gamma
    
    d = 0.0
    
    s = eta * G(c, d, *args)
    
    s, n = truncIndex(G, args, eh, s, eta, sigma, a, b, nmax, kappa)
    
    # Level-0 estimate
    
    old = sigma * h * s
    
    # add higher-level sums with refined mesh
    
    while(m <= maxlev):
        
        m = m + 1
        e2h = eh
        h = h/2
        eh = np.exp(h)
        
        s = partSum(G, args, eh, e2h, n, eta, sigma, a, b)
        
        # m-level estimate
        
        val = old/2.0 + sigma * h * s
        
        # convergence check
        if (np.abs(val-old) < eps * np.abs(val)):
            break
        
        old = val
        n = 2 * n
        
    return val


def mixedQuad(G, args, a, eps, maxlev):
    ''' This function computes the integral of function G over a semi-infinite interval [a, infty)
        using an automatic mixed quadrature rule
        according to Michalski&Mosig's paper in
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additinal parameters to be passed to `G`
        a:      lower limit of the bounded integral
        eps:    square root of the desired precision
        maxlev: order of the tanh-sinh integration 
    '''
    # The following values are recommended by Michalski&Mosig:
    
    eta = 1
    
    kappa = 1.0e-15
    
    nmax = 24
    
    # initial mesh size
    h = 1.0
        
    val = 0.0+1j*0.0
    
    # dummy variable
    sigma = None
    
    m = 0

    eh  = np.exp(h)
    
    delta = np.exp(-1.0)
    w = 2.0 * delta

    
    c = a
    d = delta
    
    s = w * G(c, d, *args)
    
    s, n1 = truncIndex(G, args,     eh, s, eta, sigma, a, None, nmax, kappa)
    s, n2 = truncIndex(G, args, 1.0/eh, s, eta, sigma, a, None, nmax, kappa)
    
    # Level-0 estimate
    
    old = h * s
    
    # add higher-level sums with refined mesh
    
    while(m <= maxlev):
        
        m = m + 1
        e2h = eh
        h = h/2
        eh = np.exp(h)
        
        s1 = partSum(G, args,     eh,     e2h, n1, eta, sigma, a, None)
        s2 = partSum(G, args, 1.0/eh, 1.0/e2h, n2, eta, sigma, a, None)        
        
        # m-level estimate
        
        val = old/2.0 + h * (s1 + s2)
        
        # convergence check
        if (np.abs(val-old) < eps * np.abs(val)):
            break
        
        old = val
        n1 = 2 * n1
        n2 = 2 * n2
        
    return val



def term(G, args, ekh, eta, sigma, a, b):
    ''' kth term of quadrature rule
        according to Michalski&Mosig's automatic tanh-sinh quadrature
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additional parameters to be passed to `G`
        ekh:    exponent of the kth term exp(kh)
        eta:    positive parameter close to unity
        sigma:  half of the integration interval
        a:      lower limit of the bounded integration interval
        b:      upper limit of the bounded integration interval    
    '''
    if b is not None:
        
        q = np.exp(-eta * (ekh - 1.0/ekh))
        
        delta = 2.0 * q / (1.0 + q)
        
        w = eta * (ekh + 1.0/ekh) * delta/(1.0+q)
        
        c = a 
        d = sigma*delta
        f1 = G(c, d, *args)
        
        c = b 
        d = -sigma*delta
        f2 = G(c, d, *args)
     
        t = w * (f1 + f2)
        
    else:
        
        delta = np.exp(-ekh)/ekh
    
        w = (1 + ekh) * delta
        
        c = a 
        d = delta
        f = G(c, d, *args)
     
        t = w * f
        
    return t

    
def partSum(G, args, eh, e2h, n, eta, sigma, a, b):
    ''' Function computing higher level partial sum
        according to Michalski&Mosig's automatic tanh-sinh quadrature
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additional parameters to be passed to `G`
        eh:     exponent of h (initial mesh size)
        e2h:    exponent of 2h (initial mesh size)
        eta:    positive parameter close to unity
        sigma:  half of the integration interval
        a:      lower limit of the bounded integration interval
        b:      upper limit of the bounded integration interval
    '''

    ekh = eh

    s = term(G, args, ekh, eta, sigma, a, b)
    
    for k in range(2, n+1):
        
        ekh = ekh * e2h
        
        s = s + term(G, args, ekh, eta, sigma, a, b)
        
    return s       



def truncIndex(G, args, eh, s, eta, sigma, a, b, nmax, kappa):
    ''' Function computing an initial truncation index and partial sum
        according to Michalski&Mosig's automatic tanh-sinh quadrature
        https://doi.org/10.1080/09205071.2015.1129915
        
        G:      function to be integrated
        args:   a tuple defining additional parameters to be passed to `G`
        eh:     exponent of h (initial mesh size)
        s:      initial value of the partial sum
        eta:    positive parameter close to unity
        sigma:  half of the integration interval
        a:      lower limit of the bounded integration interval
        b:      upper limit of the bounded integration interval
        nmax:   maximum number of points in the term quadrature rule
        kappa:  truncation error
    '''
    
    ekh = eh
    
    n = 0
    
    s1 = s
    
    while (n <= nmax):
        
        n = n + 1
        
        t = term(G, args, ekh, eta, sigma, a, b)
        
        s1 = s1 + t
        
        if (np.abs(t) <= kappa * np.abs(s1)):
        
            break
        
        ekh = ekh * eh
        
    return s1, n


        