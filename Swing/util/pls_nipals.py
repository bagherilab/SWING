import numpy as np
import sys

def pls_nipals(X,Y,A, preproc):
    '''From MATLAB code'''
    # AIM:         performs PLS calibration on X and Y
    # PRINCIPLE:   Uses the NIPALS algorithm to perform PLS model calibration
    # REFERENCE:   Multivariate Calibration, H. Martens, T. Naes, Wiley and
    #              sons, 1989
    #
    # INPUT:
    # X            matrix of independent variables (e.g. spectra) (n x p)
    # Y            vector of y reference values (n x 1)
    # A            number of PLS factors to consider
    # preproc      preprocessing applied to data
    #              0: no preprocessing
    #              1: column mean-centering of X and Y
    #
    # OUTPUT:
    # B            regression coefficients (p x 1)
    # W            X-weights (p x A)
    # T            scores (n x A)
    # P            X-loadings (p x A)
    # Q            Y-loadings (A x 1)
    # R2X          percentage of X variance explained by each PLS factor
    # R2Y          percentage of Y-variance explained by each PLS factor
    #
    # AUTHOR:      Xavier Capron
    # 			    Copyright(c) 2004 for ChemoAC
    # 			    FABI, Vrije Universiteit Brussel
    # 			    Laarbeeklaan 103, 1090 Jette
    # 			    Belgium
    #
    # VERSION: 1.0 (24/11/2004)


    n,p = X.shape
    '''From matlab code, but not used. No reference to 'center' found
    if preproc==1
        [X,mX]=center(X)
        [Y,mY]=center(Y)
    '''
    # Cast arrays as numpy matrices so that the algorithm has the same readability as the original MATLAB code
    X = np.matrix(X)
    Y = np.matrix(Y)

    # Calculate sum of squares
    ssqX = np.sum(np.power(X,2))
    ssqY = np.sum(np.power(Y,2))

    # Initialize matrices that will be returned. Not part of original code
    W = np.matrix(np.zeros([p,A]))
    t = np.matrix(np.zeros([n,A]))
    P = np.matrix(np.zeros([p,A]))
    Q = np.matrix(np.zeros([A,1]))
    R2X = np.matrix(np.zeros([A,1]))
    R2Y = np.matrix(np.zeros([A,1]))

    # Can probably be linearized
    for a in range(A):
        W[:,a] = X.T*Y
        W[:,a] = W[:,a]/np.linalg.norm(W[:,a])
        t[:,a]=np.dot(X,W[:,a])
        P[:,a]=np.dot(X.T,t[:,a])/np.dot(t[:,a].T,t[:,a])
        Q[a,0]=np.dot(Y.T,t[:,a])/np.dot(t[:,a].T,t[:,a])
        X=X-t[:,a]*P[:,a].T
        Y=Y-np.dot(t[:,a],Q[a,0])
        R2X[a,0]=(t[:,a].T*t[:,a])*(P[:,a].T*P[:,a])/ssqX*100
        R2Y[a,0]=(t[:,a].T*t[:,a])*(Q[a,0].T*Q[a,0])/ssqY*100

    Wstar=W*np.linalg.matrix_power(P.T*W,-1)
    B=Wstar*Q

    return B,Wstar, t, P, Q, R2X, R2Y, W

def vipp(x, y, t, w):

    """
    From original MATLAB code
    See https://code.google.com/p/carspls/

    #+++ vip=vipp(x,y,t,w);
    #+++ t: scores, which can be obtained by pls_nipals.m
    #+++ w: weight, which can be obtained by pls_nipals.m
    #+++ to calculate the vip for each variable to the response;
    #+++ vip=sqrt(p*q/s);
    """
    #initializing
    [p, h] = w.shape
    co = np.matrix(np.zeros([1, h]))

    # Calculate s
    for ii in range(h):
        corr = np.corrcoef(y, t[:, ii], rowvar=0)
        co[0, ii] = corr[0, 1]**2
    s = np.sum(co)

    # Calculate q
    # This has been linearized to replace the original nested for loop
    w_power = np.power(w, 2)
    d = np.multiply(w_power, co)
    q = np.sum(d, 1)
    vip = np.sqrt(p*q/s)
    return vip