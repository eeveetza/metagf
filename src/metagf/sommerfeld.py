
from textwrap import dedent

import numpy as np

import scipy.special as spec

import toms468 as patterson


def it1_s0(G, rn, b1, b2, nt1, args):
    """
    it1_s0: computes the integral T1 (0,b1) for the evaluation of zero order Sommerfeld transformation.
    The integral is computed following an elliptic contour path

    y  = it1_s0(G, rn, b1, b2, nt1, args)

    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)

    rn : float
        Normalized spatial variable (rn = k0*r)
    b1 : float
        Smaller semi-axis of the elliptic path
    b2 : float
        Larger axis of the elliptic path
    nt1 : float
        Number of integration points
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y1 : float 
        The value of integral T1 (0,b1) in the zero order Sommerfeld transformation
        along the elliptic contour
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    03OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """
    
    if not isinstance(args, tuple):
        args = (args,)
    
    #  Compute center of the ellipse.

    x0 = b2/2.0
    
    #  Compute larger semi-axis of the ellipse.
    
    aa = b2/2.0
    
    #  Compute integration limits.
    
    t1 = np.pi
    
    t2 = 0.0
    
    #  Compute weights and abcissas for Gauss integration.
    
    wt, tt = gauleg(nt1, t1, t2)
    
    #  Compute the integral.
    
    y1 = np.complex128(0.0+1j*0.0)
    
    for r in range(1, nt1+1):
        
        x = x0 + aa * np.cos(tt[r-1])
        
        y = b1 * np.sin(tt[r-1])
        
        dx = -aa * np.sin(tt[r-1])
        
        dy = b1 * np.cos(tt[r-1])
        
        w = x + 1j*y
        
        dw = dx + 1j*dy
        
        kro = w
        
        y1 = y1 + wt[r-1] * spec.jv(0, w*rn) * w * G(kro, *args) * dw
        
    
    return y1


def it2_s0(G, rn, b2, nt2, y1, cs, prec, nit, args):
    """
    it2_s0: computes the integral T2 (b2-infty) for the evaluation of zero order Sommerfeld transformation
            The integral is computed using the weighted averages method

    y  = it2_s0(G, arg, rn, b2, nt2, y1, cs, prec)


    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)

    rn : float
        Normalized spatial variable (rn = k0*r)
    b2 : float
        Starting point of the integration path 
        (corresponds to the larger ellipse axis)
    nt2 : integer
        Number of integration points per segment
    y1 : complex
        Value of the integral along the ellipse
    cs : integer
        cs = 1 when source and obesrver are at the same z-coordinate
    prec : float
        Required precision in the WA algorithm
    nit : integer
        Maximum number of iterations
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y2 : float 
        The value of integral T2 (b2, infty) in the zero order Sommerfeld transformation
        using weigthed averages
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    03OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """
    if (cs == 1):
        
        uu = -1.00      # Convergence factor same interfaces.
    else:
        
        uu = -3.00;     # Convergence factor different interfaces.
    
    
    #  Compute integration limits.
    
    aa = np.zeros(nit+1, dtype = 'double')
    
    aa[0] = b2
    
    for l in range(2, nit + 2):
        
         aa[l-1] = aa[l-2] + np.pi/rn
        
    '''    if (rn <= 1e-5 and z <= 1e-5):
            
            aa[l-1] = aa[l-2] + np.pi/1e-3
            
        elif (rn <= 1e-5):
            
            aa[l-1] = aa[l-2] + np.pi/z
        
        else:
            aa[l-1] = aa[l-2] + np.pi/rn
    '''
    #  Weighted average algorithm.
    
    err = 1.0
    
    l = 0
    
    ss = np.zeros((2, nit), dtype = 'complex128')
    
    ss[1,0] = 0.0+1j*0.0

    
    while (err > prec and  l < nit-1):
        
        l = l + 1
        
        for  n in range(1, l + 1):
            
            #Shift back previous estimation.
            
            ss[0,n-1] = ss[1,n-1]
            
            if (n == 1):

                ss[1,n-1] = intt2_s0(G, rn, nt2, aa[l-1], aa[l], args) + ss[1,n-1]
            else:
                
                #Compute new estimation.
                
                try:
                
                    wm = np.power(aa[l], (uu-n+2))
                    
                    wp = np.power(aa[l-1], (uu-n+2))
                    
                    ss[1,n-1] = (wm*ss[0,n-2] + wp*ss[1,n-2])/(wm+wp)
                    
                        
                except RuntimeWarning:
                    return ss[1,n-2]
        
            
            
        
        if (l > 1):
            
            prev = np.abs(y1 + ss[1,l-2])
            
            actu = np.abs(y1 + ss[1,l-1])
            
            if (np.isnan(actu) or np.isinf(actu)):
                return ss[1,l-3]
            
            if (actu > 5.0):
                
                # Check with relative error.
                
                err = np.abs(actu-prev)/actu
            else:
                
                #Check with absolute error.
                
                err = np.abs(actu-prev)
            
        
    if (err  > 0.01):
        print(dedent('''
        ===============================================
                Can not get enough accuracy in
                      weighted average.
                       S0 interaction.
                   Error = ,  {err}
        
                   Continuing Execution.
        ===============================================
        ''' ))
    
    
    y2 = ss[1,l-1]
    
    return y2


def intt2_s0(G, rn, nt, a, b, args):
    """
    intt2_s0: computes the integral of the following function
                          J0(x*r)*x*G(x) 
              in the bounded interval (a,b), where G(x, args) is spectral domain function
              and J0(x) is Bessel function of zero order and first kind

    y  = intt2_s0(G, rn, nt, a, b, args)


    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)
    rn : float
        Normalized spatial variable (rn = k0*r)
    nt : integer
        Number of points in Gauss-Legendre integration
    a : float
        Lower integration limit
    b : float
        Upper integration limit
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y : float 
        The value of integral in the zero order Sommerfeld transformation
        over a bounded interval (a, b)

    
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    03OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """
    
    #  Compute weights and abcissas for Gauss integration.
    
    wx, xx = gauleg(nt, a, b)
    
    
    #  Compute the integral.

    
    y = 0.0+1j*0.0
    
    for r in range(1, nt+1):
        
        #Compute spectral domain Green's function.
        kro = xx[r-1]
        fsd = G(kro, *args)
        
        y = y + wx[r-1] * spec.jv(0, xx[r-1] * rn) * xx[r-1] * fsd
     
    
    return y


def it1_s1(G, rn, b1, b2, nt1, args):
    """
    it1_s1: computes the integral T1 (0,b1) for the evaluation of first order Sommerfeld transformation.
    The integral is computed following an elliptic contour path

    y  = it1_s1(G, rn, b1, b2, nt1)

    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)

    rn : float
        Normalized spatial variable (rn = k0*r)
    b1 : float
        Smaller semi-axis of the elliptic path
    b2 : float
        Larger axis of the elliptic path
    nt1 : float
        Number of integration points
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y1 : float 
        The value of integral T1 (0,b1) in the first order Sommerfeld transformation
        along the elliptic contour
    
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    11OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """
    
    #  Compute center of the ellipse.

    x0 = b2/2.0
    
    #  Compute larger semi-axis of the ellipse.
    
    aa = b2/2.0
    
    #  Compute integration limits.
    
    t1 = np.pi
    
    t2 = 0.0
    
    #  Compute weights and abcissas for Gauss integration.
    
    wt, tt = gauleg(nt1, t1, t2)
    
    #  Compute the integral.
    
    y1 = np.complex128(0.0+1j*0.0)
    
    for r in range(1, nt1+1):
        
        x = x0 + aa * np.cos(tt[r-1])
        
        y = b1 * np.sin(tt[r-1])
        
        dx = -aa * np.sin(tt[r-1])
        
        dy = b1 * np.cos(tt[r-1])
        
        w = x + 1j*y
        
        dw = dx + 1j*dy
        
        kro = w
        
        fsd = G(kro, *args)
        
        y1 = y1 + wt[r-1] * spec.jv(1, w*rn) * w * w * fsd * dw
        
    
    return y1


def it2_s1(G, rn, b2, nt2, y1, cs, prec, nit, args):
    """
    it2_s1: computes the integral T2 (b2-infty) for the evaluation of first order Sommerfeld transformation
            The integral is computed using the weighted averages method

    y  = it2_s1(G, rn, b2, nt2, y1, cs, prec)

    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)

    rn : float
        Normalized spatial variable (rn = k0*r)
    b2 : float
        Starting point of the integration path 
        (corresponds to the larger ellipse axis)
    nt2 : integer
        Number of integration points per segment
    y1 : complex
        Value of the integral along the ellipse
    cs : integer
        cs = 1 when source and obesrver are at the same z-coordinate
    prec : float
        Required precision in the WA algorithm
    nit : integer
        Maximum number of iterations
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y2 : float 
        The value of integral T2 (b2, infty) in the first order Sommerfeld transformation
        using weigthed averages
    
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    11OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """
    if (cs == 1):
        
        uu = -1.00      # Convergence factor same interfaces.
    else:
        
        uu = -3.00;     # Convergence factor different interfaces.
    
    #z = abs(arg.z)
    
    #  Compute integration limits.
    
    aa = np.zeros(nit+1, dtype = 'double')
    
    aa[0] = b2
    
   
    for l in range(2, nit + 2):
        
         aa[l-1] = aa[l-2] + np.pi/(rn)
    '''    
        if (rn <= 1e-3 and z <= 1e-3):
            
            aa[l-1] = aa[l-2] + np.pi/1e-3
            
        elif (rn <= 1e-3):
            
            aa[l-1] = aa[l-2] + np.pi/z
        
        else:
            aa[l-1] = aa[l-2] + np.pi/rn
    
    '''
    #  Weighted average algorithm.
    
    err = 1.0
    
    l = 0
    
    ss = np.zeros((2, nit), dtype = 'complex128')
    
    ss[1,0] = 0.0+1j*0.0
    
    while (err > prec and  l < nit-1):
        
        l = l + 1
        
        for  n in range(1, l + 1):
            
            #Shift back previous estimation.
            
            ss[0,n-1] = ss[1,n-1]
            
            if (n == 1):
                
                ss[1,n-1] = intt2_s1(G, rn, nt2, aa[l-1], aa[l], args) + ss[1,n-1]
            else:
                
                #Compute new estimation.
                
                try:
                
                    wm = np.power(aa[l], (uu-n+2))
                    
                    wp = np.power(aa[l-1], (uu-n+2))
                    
                    ss[1,n-1] = (wm*ss[0,n-2] + wp*ss[1,n-2])/(wm+wp)
                    
                        
                except RuntimeWarning:
                    return ss[1,n-2]
        
        
        if (l > 1):
            
            prev = np.abs(y1 + ss[1,l-2])
            
            actu = np.abs(y1 + ss[1,l-1])
            
            if (np.isnan(actu) or np.isinf(actu)):
                return ss[1,l-3]
            
            if (actu > 5.0):
                
                # Check with relative error.
                
                err = np.abs(actu-prev)/actu
            else:
                
                #Check with absolute error.
                
                err = np.abs(actu-prev)
            
        
    if (err  > 0.01):
        print(dedent('''
        ===============================================
                Can not get enough accuracy in
                      weighted average.
                       S0 interaction.
                   Error = ,  {err}
        
                   Continuing Execution.
        ===============================================
        ''' )), err
    
    
    y2 = ss[1,l-1]
    
    return y2


def intt2_s1(G, rn, nt, a, b, args):
    """
    intt2_s1: computes the integral of the following function
                          J1(x*r)*x^2*G(x,*args) 
              in the bounded interval (a,b), where G(x,*args) is spectral domain function
              and J1(x) is Bessel function of first order and first kind

    y  = intt2_s1(G, arg, rn, nt, a, b)


    Parameters
    ----------

    G : function
        Spectral domain function G(kro, *args)
    rn : float
        Normalized spatial variable (rn = k0*r)
    nt : integer
        Number of points in Gauss-Legendre integration
    a : float
        Lower integration limit
    b : float
        Upper integration limit
    args : tuple, optional
        Extra arguments to pass to `G`
        
    Returns
    -------
    y : float 
        The value of integral in the zero order Sommerfeld transformation
        over a bounded interval (a, b)
    
        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    11OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      
    
    """
    
    #  Compute weights and abcissas for Gauss integration.
    
    wx, xx = gauleg(nt, a, b)
    
    #  Compute the integral.
    
    y = 0.0+1j*0.0
    
    for r in range(1, nt+1):
        
        #Compute spectral domain Green's function.
        
        kro = xx[r-1]
        
        fsd = G(kro, *args)
        
        y = y + wx[r-1] * spec.jv(1, xx[r-1] * rn) * xx[r-1]*xx[r-1] * fsd
        

    return y

    

def gauleg(n, a1, a2):
    """
    gauleg: Gauss-Legendre n-point quadrature

    w, x  = gauleg(n, a1, a2)

    where:    Units,  Definition                             Limits
    n:        -       Number of integration points           >1
    a1:       -       Lower limit of the integration
    a2:       -       Upper limit of the integration
    w:        -       Vector of weights
    x:        -       Vector of abscissae
    
    
    This subroutine computes the weights and abcissae for a Gauss- Legendre 
    n-point quadrature formula.
    The program computes the n-roots of the Legendre polynomial of order n.
    It uses a Newton method to find numerically the zeros.

        
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    03OCT19     Ivica Stevanovic, OFCOM         First implementation in python
                                                      from A. A. Melcon Fortran routine
    
    """

    eps = 1.0e-15
    x = np.zeros(n, dtype = np.float64)
    w = np.zeros(n, dtype = np.float64)
    

    m = int(np.floor((n+1.0)/2.0))          # Due to symmetry only half need to be computed.
    xm = 0.5*(a2+a1)         # Mapping from (a1,a2) to (-1,1)
    xl = 0.5*(a2-a1)

    z1 = -1.0                # Store the previous zero.

    for i in range(1, m+1):
    
        #   Good aproximation for the i-th zero of Legendre Polynomial of order n.
        z = np.cos(np.pi*(i-0.25)/(n+0.5))
        
        while (abs(z-z1) > eps):        # Newton method iterations.
            
            p1 = 1.0    # Initializing P-1(x).
            p2 = 0.0    # Initializing P0(x).
            
            # Computing with recurrence algorithm the Legendre
            # polynomial of order n, evaluated in z.
            for k in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2.0*k-1.0)*z*p2-(k-1.0)*p3)/k
            
            
            pp = n*(z*p1-p2)/(z*z-1.0);  # This expression is the derivative.
            z1 = z                       # Saving the previous zero.
            z = z1-p1/pp                # Computing the new zero with Newton method.
        
        
        x[i-1] = xm - xl*z                   # Mapping from (-1,1) to (a1,a2).
        x[n-i] = xm + xl*z                   # Placing the symmetric point.
        w[i-1] = 2.0/((1.0-z*z)*pp*pp)       # computation of weights.
        
        # This constant is needed for mapping from (a1,a2) to
        # the interval (-1,1), in the weights.
        
        w[i-1] = w[i-1]*xl
        w[n-i] = w[i-1]                     # Placing the symmetric weight.


    return w, x 

def halfsine(func, a, c, nints, args):
    """
    half_sine: computes the integral of the function func(x,args) 
    over a half-sine contour defined by the widht `a` and height `c` using Gauss-Legendre
    quadrature with required number of integration points `nints`
    y  = half_sine(func, a, c, nints, args)

       
    Parameters
    ----------

    func : function
        A Python function or method. If `func`takes many 
        arguments, it is integrated along the axis corresponding to the
        first argument.
    a : float
        width of the half-sine contour
    c : float
        maximum height of the half-sine contour     
    nints: integer
        number of integration points
    args : tuple, optional
        Extra arguments to pass to `func`
    

    Returns
    -------
    y : float
        The value of the integral
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    14NOV19     Ivica Stevanovic, OFCOM         First implementation
                                              
    
    """
    
    def integrand(x, func, a, c, args):
        
        """
        This subroutine defines an integrand function required for the
        computation of inverse Sommerfeld integral over the bounded interval
        (0, a) along the half-sine contour defined by the width a and 
        the height c
        
        Parameters
        ----------
        x : float
            Parameter over which the integration is performed
        func : function
            A Python function or method. 
        a : float
            width of the half-sine contour
        c : float
            maximum height of the half-sine contour     
        args : tuple, optional
            Extra arguments to pass to `func`
        
    
        Returns
        -------
        result : float
            the integrand required for the computation of the inverse Sommerfeld
            integral over the bounded interval (0, a) along the half-sine contour.
        
        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v1    10NOV19     Ivica Stevanovic, OFCOM         First implementation

        """
                                                         
    
        
        targ = np.pi * (1.0 - x) / 2.0 
        
        kro = a / 2.0 * (1 + np.cos( targ) ) + 1j* c * np.sin( targ )
        
        jcb = a * np.pi/4.0 * np.sin( targ ) - 1j * c * np.pi/2.0 * np.cos( targ )
        
        result = func(kro, *args) * jcb
        
        return result
    
    # end integrand0
    
    wx, xx = gauleg(nints, -1.0, 1.0)
    
    #  Compute the integral.
    
    y = 0.0+1j*0.0
    
    for r in range(1, nints+1):
        
        #Compute spectral domain Green's function.
        
        t = xx[r-1]
        
        fsd = integrand(t, func, a, c, args)
        
        y = y + wx[r-1] * fsd
        

    return y
    
def quad(func, a, b, nints, args):
    """
    quad: computes the integral of the function func(x,args) 
    over real-axis x using Gauss-Legendre 
    quadrature with required number of integration points `nints`
    over interval  (a,b)
    y  = quad(func, a, b, nints, args)

       
    Parameters
    ----------

    func : function
        A Python function or method. If `func`takes many 
        arguments, it is integrated along the axis corresponding to the
        first argument.
    a : float
        lower limit of the integration interval
    b : float
        upper limit of the integration interval   
    nints: integer
        number of integration points
    args : tuple, optional
        Extra arguments to pass to `func`
    

    Returns
    -------
    y : float
        The value of the integral
    
    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v1    14NOV19     Ivica Stevanovic, OFCOM         First implementation
                                              
    
    """

    
    # end integrand0
    
    wx, xx = gauleg(nints, a, b)
    
    #  Compute the integral.
    
    y = 0.0+1j*0.0
    
    for r in range(1, nints+1):
        
        #Compute spectral domain Green's function.
        
        y = y + wx[r-1] * func(xx[r-1], *args)
        

    return y
