"""Exploratory, ungated sanity check for docs/design/implicit-exponential-hermite.md.

Frozen-field 1D1V AW operator (single cosine field, |k|<=K, Nn Hermite orders, same operator family as
studies/highk_eig.py) and the per-step growth rate of one-step maps: RK4, Lawson-RK4, implicit midpoint, backward
Euler, Strang with exact streaming and exact (nilpotent) force flows. The first run was made before this file was
committed; studies/imex/frozen_toy.txt is the record from the committed file. Not a declared-gate study.
"""
import numpy as np, scipy.linalg as sl
K,N=5,64; a=np.sqrt(2e-3); E1=4e-3; qm=-1.0
ks=np.arange(-K,K+1); nk=len(ks); idx={k:i for i,k in enumerate(ks)}
def build(E1):
    n=np.arange(N)
    T=np.zeros((N,N))
    for m in range(N-1): T[m,m+1]=T[m+1,m]=np.sqrt(m+1)/np.sqrt(2)
    D=np.zeros((N,N))
    for m in range(1,N): D[m,m-1]=np.sqrt(2*m)/a
    S=np.zeros((nk,nk),complex)   # streaming wavenumber diag
    Ef=np.zeros((nk,nk))          # convolution by E=E1 cos(x) -> modes +-1
    for i,k in enumerate(ks):
        for s in (-1,1):
            if k+s in idx: Ef[i,idx[k+s]]=E1/2
    Ls=np.kron(np.diag(-1j*ks*a),T).astype(complex)   # index (k,n)
    Lf=qm*np.kron(Ef,D)
    return Ls,Lf,T,D
Ls,Lf,T,D=build(E1); L=Ls+Lf
def rate(M,h):
    w=np.linalg.eigvals(M); return np.log(np.abs(w).max())/h
ev=np.linalg.eigvals(L); print("exact max Re lambda",ev.real.max(), "stream rho",np.abs(np.linalg.eigvals(Ls)).max(),"force norm",np.linalg.norm(Lf,2))
# check nilpotent force: Lf^N per k mixing? D nilpotent => (Ef kron D)^N=0
print("force nilpotent:",np.abs(np.linalg.matrix_power(Lf/ (abs(qm)*E1), N)).max())
I=np.eye(L.shape[0])
def rk4(A):
    return I+A+A@A/2+A@A@A/6+A@A@A@A/24
# Lawson-RK4: u+ = e^{hLs} * RK4 applied to interaction-picture; build via Lawson formulas
def lawson4(h):
    E=sl.expm(h*Ls); E2=sl.expm(h*Ls/2)
    F=lambda X:Lf@X
    u=I
    k1=F(u); k2=F(E2@(u+h/2*k1)); k3=F(E2@u+h/2*k2); k4=F(E@u+h*E2@k3)
    return E@u+h/6*(E@k1+2*E2@(k2+k3)+k4)
def be(h): return np.linalg.inv(I-h*L)
def mid(h): return np.linalg.solve(I-h/2*L,I+h/2*L)
def strang(h):
    Es=sl.expm(h/2*Ls); Ff=sl.expm(h*Lf); return Es@Ff@Es
print("exact gamma", ev.real.max())
for h in [0.01,0.05,0.2,1,3]:
    row=[h]
    for f in (lambda h:rk4(h*L),lawson4,mid,be,strang):
        try: row.append(rate(f(h),h))
        except Exception as e: row.append(np.nan)
    print("h=%.2f RK4 %.3g Lawson4 %.3g midpoint %.3g BE %.3g Strang(exact streaming,exact force) %.3g"%tuple(row))
print("RK4 imaginary-axis limit h*rho:",2.8, "-> h_max ~",2.8/np.abs(np.linalg.eigvals(Ls)).max())
