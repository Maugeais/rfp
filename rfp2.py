 
#!/usr/bin/python3

from math import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib, numpy.linalg

# Avec ou sans partie constante !

dec = 1

# Definition de la classe polynome
# Range dans un tableau par ordre croissant
class pol:
    def __init__(self, coef):
        self.deg = len(coef)-1
        self.coef = np.array(coef, dtype = complex)

    # Evaluation a l'aide du schema de Horner
    def eval(self, x):
        y = self.coef[self.deg]+0*x
        for i in range(self.deg-1, -1, -1):
            y = y*x+self.coef[i]

        return(y)

    # Affichage
    def __str__(self):

        if (self.deg == 0):
            poly = str(self.coef[0])
            return(poly)

        poly = str(self.coef[0])+'+'
        i = 0

        for i in range(1, self.deg):
            poly += str(self.coef[i])+'*X^'+str(i)+'+'

        poly += str(self.coef[-1])+'X^'+str(i+1)

        return(poly)

    # Addition de polynomes
    def __add__(self, P):
        Q = np.zeros(max(self.deg, P.deg)+1, dtype = complex)
        for i in range(self.deg+1):
            Q[i] = self.coef[i]

        for i in range(P.deg+1):
            Q[i] += P.coef[i]

        return(pol(Q))

    def __sub__(self, P):
        Q = np.zeros(max(self.deg, P.deg)+1, dtype = complex)
        for i in range(self.deg+1):
            Q[i] = self.coef[i]

        for i in range(P.deg+1):
            Q[i] -= P.coef[i]

        return(pol(Q))
    
    def __mul__(self, P):
        Q = np.zeros(self.deg+P.deg+1, dtype = complex)
        
        for i in range(self.deg+1) :
            for j in range(P.deg+1) :
                Q[i+j] += self.coef[i]*P.coef[j]
        
        return(pol(Q))


    # Multiplication par une constante
    def __rmul__(self, l):
        Q = l*self.coef
        return(pol(Q))

    # Calcul du produit scalaire
    def scal(self, Q, omega, h):
        r = 0
        for i in range(len(omega)):
            r += abs(h[i])**2*self.eval(omega[i])*np.conj(Q.eval(omega[i]))
            
        return(r)

    # Calcul la multplication par x
    def dec(self):
        Q = np.zeros(self.deg+2, dtype = complex)
        for i in range(self.deg+1):
            Q[i+1]=self.coef[i]
        return(pol(Q))

    def norm(self, omega, h):
        r = np.sqrt(abs(self.scal(self, omega, h)))
        return(r)
        
    def der(self):
        Q = np.zeros(self.deg, dtype = complex)
        for i in range(self.deg):
            Q[i] = (i+1)*self.coef[i+1]
            
        return(pol(Q))
        

# Calcule une base orthonormee par l'algo de Forsythe
# cf. Generation and use of orthogonal polynomials for data-fitting with a digital computer
def orthogonal(N, omega, h):
    base = []

    # Initialisation
    P0 = pol([1])
    P0 = 1/(P0.norm(omega, h))*P0


    base.append(P0)

    u = P0.dec().scal(P0, omega, h)
    P1 = P0.dec()-u*P0
    if (P1.norm(omega, h) != 0) :
        P1 = 1/(P1.norm(omega, h))*P1

    base.append(P1)

    # Recurrence
    for i in range(2, N):
        u = P1.dec().scal(P1, omega, h)
        v = P1.dec().scal(P0, omega, h)

        P = P1.dec()-u*P1-v*P0
        if (P.norm(omega, h) != 0) :
            P = 1/(P.norm(omega, h))*P

        P0 = P1
        P1 = P

        base.append(P)

    return(base)

# Analyse globale

def modan(n, omega, h, sym = False, spurious = False, disp = False, origin = True):
    """ Analyse globale 
    m est le degré du numérateur C, n le degré du dénominateur """
    
    if (sym) :
        omega = np.concatenate((-omega, omega))
        h = np.concatenate((np.conj(h), h))
        n = 2*n
    
    m = n - origin - dec
    
    # Si origin, on divise l'impédance par omega
    if origin :
        h /= omega
        
    Bphi = orthogonal(m+1, omega, np.ones(len(omega)))
    Btheta = orthogonal(n+1, omega, h)
   
    # Creation des matrices P et T
    P = np.matrix(np.zeros((len(omega), m+1), dtype = complex))
    T = np.matrix(np.zeros((len(omega), n), dtype = complex))
    W = np.zeros((len(omega), 1), dtype = complex)

    for i in range(len(omega)):
        for j in range(m+1):
            P[i, j] = Bphi[j].eval(omega[i])
        for j in range(n):
            T[i, j] = h[i]*Btheta[j].eval(omega[i])

        W[i] = h[i]*Btheta[-1].eval(omega[i])

    # Calcul des coefficients dans la base des pol orthogonaux

    X = P.H*T
    H = P.H*W
    I = np.identity(n, dtype = complex)
    D = np.linalg.solve(I-X.H*X, X.H*H)
    C = P.H*W+P.H*(T*D)

    C = np.resize(C, m+1)
    D = np.resize(D, n+1)
    D[-1] = 1

    # Calcul es coefficients dans la base des monomes

    Cp = C[0]*Bphi[0]
    for i in range(1, m+1):
        Cp = Cp + C[i]*Bphi[i]

    Dp = D[0]*Btheta[0]
    for i in range(1, n+1):
        Dp = Dp + D[i]*Btheta[i]

    # Si origin, il faut remultiplier h par omega pour le calcul d'erreur, et multiplier le numérateur par x
    if origin :
        h *= omega
        Cp = pol([0]+list(Cp.coef))
        
    hp =  Cp.eval(omega)/(Dp.eval(omega))
        
    if not spurious or sym :
        p = np.roots(Dp.coef[::-1])
        I = np.where(np.real(p) > 0)[0]
        
        #computeResidues(p, omega, h)    
        
             
        #Cp = pol(np.polyfit(omega, hp*Dp.eval(omega), len(I)-1))
        #Cp.coef *= norm/Cp.coef[-1]
        
                
           
    # Calcul de l'erreur
    
    hp =  Cp.eval(omega)/(Dp.eval(omega))
        
    e = np.linalg.norm(hp-h)/np.linalg.norm(h)
    
    if (disp) :
    
        plt.plot(omega, np.abs(h))              
        plt.plot(omega, np.abs(hp))              
        plt.show()

    return(Cp, Dp, e)


def estimatePoles(omega, h) :
    """ Estimation du nombre de pôles.
    Il faut que l'impédance soit lissée, et donc en dehors du bruit..."""
    
    # C'est le nombre de maximum 
    # c'est qunad il y a changement de signe h(i) > h(i+1) mais h(i-1) < h(i)
    
    I = np.logical_and(h[2:]<h[1:-1], h[1:-1]>h[:-2])
    
    Max = [omega[i+1] for i, x in enumerate(I) if x]                              
        
    return(Max)

def translateResiduesToZero(p, omega, h) :
    
    A = np.zeros((len(p), len(p)), dtype = complex)

    B = np.zeros(len(p), dtype = complex)
    for j in range(len(p)) :
        # Find the closest omega_j
        jo = (np.abs(omega - np.real(p[j]))).argmin()
        B[j] = h[jo]
        for i in range(len(p)) :
            A[j, i] = omega[jo]/((omega[jo]-p[i])*(omega[jo]+np.conj(p[i])))
            
    # C'est ici qu'on fait l'approximation de'impédance nulle en 0        
    alpha = 1j*np.imag(np.linalg.solve(A, B))
    
    r = alpha*p/(2*np.real(p))
            
    return(r)
    

from scipy.signal import residue
def computePoles(C, D, sym = False, spurious = True) :
    """ Calcul poles et residus de la fraction C/D, on suppose que les poles sont simples !!! """

    # Calcul des residus, classe par ordre decroissant

    p = np.roots(D.coef[::-1])
    r = C.eval(p)/D.der().eval(p)
    
    # r, p, c = residue(C.coef[::-1], D.coef[::-1])
       
    I = np.argsort(np.real(p)) #[::-1]
    r = r[I]
    p = p[I]
        
    if not spurious or sym :
        I = np.where(np.real(p) > 0)[0]
        
        #p, r = updateResidues(p, I, omega, h)
        r = r[I]
        p = p[I]   
        
        #for i in range(len(p)) :
        #    r[i] = np.real(r[i])*p[i]/(np.imag(p[i]))
        
        # Il faut recalculer les résidus car on a enlevre trop de choses ???
                    
    return(r, p)

def impedanceFromCoeffs(r, p, omega, sym) :
    y = np.zeros(omega.shape, dtype = 'complex')
    
    for i in range(len(r)) :
        y += r[i]/(omega-p[i])
            
        if (sym) :
            y -= np.conj(r[i])/(omega+np.conj(p[i]))

        
    return(y)

import multiprocessing
def wrappedModan(args) :
        
        _, _, eps = modan(*args)
        
        return(eps)
    
def nbModes(Nmin, Nmax, omega, h, sym = True, display = False, ech=1):
    
    norm2 = np.linalg.norm(h)
    
    
    if (sym) :
        omegaSym = np.concatenate((-omega[::ech], omega[::ech]))
        hSym = np.concatenate((np.conj(h[::ech]), h[::ech]))
    else:
        omegaSym = omega[::ech]
        hSym = h[::ech]


    epsMin = 1e10
    
    Rref = []
    Pref = []
    Nmin = max(Nmin, 1)
    
    args = [[(sym+1)*i, omegaSym, hSym] for i in range(Nmin, Nmax+1)]
    
    
    pool = multiprocessing.Pool(4)
    residues = pool.map(wrappedModan, args)
    
    # print(Nmin, Nmax, np.argmin(out)+Nmin)
    # print(out)
    
    return(np.argmin(residues)+Nmin, residues)
