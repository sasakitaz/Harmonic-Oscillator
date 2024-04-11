"""
Hamiltonian
H = p^2/2mu + kzz r^2 + kzzzz r^4

動径方向に非調和性を持つポテンシャル
"""
import numpy as np
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt

#グローバル変数の定義
#matrix size
n3: int = 10

#molecular paramter (amu)
mu: float = 1

#potential paramter
kzz: float = 4000    #cm-1 A^-2
kzzzz:float = 100   #cm-1 AA^-4

#constant
c: float = 2.998*10**8 #speed of light
hbar: float = 1.05457*10**(-34)    #Planck constant
NA: float = 6.02*10**(23)  #Avogadro constant
mu = mu/NA/10**3    #kg
kzz_si: float = kzz*1.98645*10**(-23)*10**(20)    #J m-2
hbaromega: float = np.sqrt(2*kzz_si/mu)/c/100     #cm-1

def Factorial(n):
    if n < 0:
        return -100
    elif n == 0 or n == 1:
        return 1
    elif n == 1/2:
        return np.sqrt(np.pi)/2
    else:
        return n*Factorial(n - 1) 

def DoubleFactorial(n):
    if n < 0 :
        return -100
    elif n == 0 or n == 1:
        return 1
    else:
        return n*DoubleFactorial(n - 2) 

#calculation of matrixelement
class MatrixElement:
    def __init__(self, nrow, lrow, mrow, ncolumn, lcolumn, mcolumn):
        self.nrow: int = nrow
        self.lrow: int = lrow
        self.mrow: int = mrow
        self.qrow: int = 1/2*(nrow - lrow)
        self.ncolumn: int = ncolumn
        self.lcolumn: int = lcolumn
        self.mcolumn: int = mcolumn
        self.qcolumn: int = 1/2*(ncolumn - lcolumn)
    
    def Wigner3j_product(self, p, q):
        symbol1: float = wigner_3j( self.lrow, p, self.lcolumn,
                                   -self.mrow, q, self.mcolumn)
    
        symbol2: float = wigner_3j( self.lrow, p, self.lcolumn,
                                    0        , 0,            0)
        symbol: float = symbol1*symbol2
        return symbol
    
    def harmonic_oscillator(self):
        HO: float = 0
        if self.qrow - self.qcolumn == 0 and self.lrow - self.lcolumn == 0 and self.mrow - self.mcolumn == 0:
            HO = 2*self.qcolumn + self.lcolumn + 3/2
        else:
            HO = 0
        return HO
    
    def multipole(self, L, M, S):
        #calculation of factorial sum
        nuMax = int(self.qrow)
        coeff = 0
        for nu in range(0, nuMax + 1):
            nu1 = DoubleFactorial(L + self.lcolumn + self.lrow + 2*nu + 2*S + 1)
            nu2 = Factorial(1/2*(L + self.lrow - self.lcolumn) + S + nu)
            nu3 = Factorial(1/2*(L + self.lrow - self.lcolumn) + nu + S - self.qcolumn)
            nu4 = Factorial(nu)
            nu5 = Factorial(self.qrow - nu)
            nu6 = DoubleFactorial(2*self.lrow + 2*nu + 1)
            temp = (-1)**(nu)*(nu1*nu2)/(nu3*nu4*nu5*nu6)
            if nu1 < 0 or nu2 < 0 or nu3 < 0 or nu4 < 0 or nu5 < 0 or nu6 < 0:
                temp = 0
            coeff += temp
        
        V_multipole = ((-1)**(self.mrow + self.qcolumn)*np.sqrt((2*L + 1)/(4*np.pi))
                       *coeff
                       *np.sqrt(
                           (2**self.qcolumn*(2*self.lcolumn + 1)*(2*self.lrow + 1)*Factorial(self.qrow   )*DoubleFactorial(2*self.lrow + 2*self.qrow + 1))
                          /(2**(self.qrow + 2*S + L)                              *Factorial(self.qcolumn)*DoubleFactorial(2*self.lcolumn + 2*self.qcolumn + 1))
                          )
                       *self.Wigner3j_product(L, M)
                      )
        return V_multipole
    
    
def Hamiltonian():
    result: np.ndarray = np.array([[+ MatrixElement(nrow, lrow, mrow, ncolumn, lcolumn, mcolumn).harmonic_oscillator() *hbaromega/4
                                    + MatrixElement(nrow, lrow, mrow, ncolumn, lcolumn, mcolumn).multipole(0, 0, 2) *kzzzz
                for ncolumn in range(0, n3 + 1)               
                for lcolumn in range(ncolumn, -1, -2)
                for mcolumn in range(- lcolumn, lcolumn + 1)
            ]
            for nrow in range(0, n3 + 1)
            for lrow in range(nrow, -1, -2)
            for mrow in range(-lrow, lrow + 1)
        ])
    result = result.astype(np.float64)
    return result

#diagonarization and generation of eigen vector list
#eig_vec: raw data, N*N eigen vector matrix
#eigen_vector: the list sorted by quantum number: [energy, coefficient, n, l, m]
def diagonalization(H):
    dim: list = len(H)
    eig_val: np.ndarray = np.zeros(dim)
    eig_vec: np.ndarray = np.zeros((dim, dim))
    #diagonalization
    eig_val,eig_vec = np.linalg.eigh(H)
    
    #make the list sorted by quantum number 
    eigen_vector: list = []
    ndim: list = []
    ldim: list = []
    ndimsum: list = [0]
    ldimsum: list = [0]
    allownvib: list = []
    allowlvib: list = []
    ndimsum_count: int = 0
    ldimsum_count: int = 0
    for nvib_count in range (0, n3 + 1):
        ndim_count = 0
        for ll in range (nvib_count, -1, -2):
            allowlvib.append(ll)
            ldim.append(2*ll + 1)
            ldimsum_count += 2*ll + 1
            ldimsum.append(ldimsum_count)
            ndim_count += 2*ll + 1
        ndim.append(ndim_count)
        allownvib.append(nvib_count)
        ndimsum_count += ndim_count
        ndimsum.append(ndimsum_count)
    
    for r in range(0, len(eig_vec)):
        for c in range(0, len(eig_vec[r])):
            nsurp = c 
            for dn in range (0, len(ndimsum)):
                if ndimsum[dn] <= nsurp and nsurp < ndimsum[dn + 1]:
                    quantn: int = allownvib[dn]
            for dl in range (0, len(ldimsum)):
                if ldimsum[dl] <= nsurp and nsurp < ldimsum[dl + 1]:
                    quantl: int = allowlvib[dl]
                    quantm: int = nsurp - ldimsum[dl] - allowlvib[dl]
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantn, quantl, quantm])
    return eig_val, eigen_vector

#option
def MakeGraph_Potential(eig_val, eig_vec):
    fig = plt.figure(dpi=300, figsize=(4,3))
    ax1 = fig.add_subplot(111)
    
    #Function
    x1 = np.linspace(0, 2.0, 101)
    y1 = float(kzz)*x1**2 + float(kzzzz)*x1**4
    
    for ii in range (0, len(eig_val)):
        plt.hlines(eig_val[ii], -0.5, -0.1, color="k")
    
    #Eigen state
    for ii in range (0, len(eig_vec)):
        if abs(eig_vec[ii][1]) > 0.5:
            if abs(eig_vec[ii][3]) == 0:
                plt.hlines(eig_vec[ii][0], 0, 0.4, color="k")
            elif abs(eig_vec[ii][3]) == 1:
                plt.hlines(eig_vec[ii][0], 0.5, 0.9, color="k")
            elif abs(eig_vec[ii][3]) == 2:
                plt.hlines(eig_vec[ii][0], 1.0, 1.4, color="k")
            elif abs(eig_vec[ii][3]) == 3:
                plt.hlines(eig_vec[ii][0], 1.5, 1.9, color="k")
    ax1.text(0.7, 0.85, "$k_{zz}$ = " + str(kzz) + "\n$k_{zzzz}$ = " + str(kzzzz), fontsize = "11", transform = ax1.transAxes)
    ax1.text(-0.5, 0, "all")
    ax1.text( 0.0, 0, "$l = 0$(s)")
    ax1.text( 0.5, 0, "$l = 1$(p)")
    ax1.text( 1.0, 0, "$l = 2$(d)")
    ax1.text( 1.5, 0, "$l = 3$(f)")
    fig.suptitle("Radial potential of 3D oscillator")
    ax1.plot(x1, y1)
    ax1.set_ylim(-500, 10000)
    ax1.set_xlabel('radial/ $\mathrm{\AA}$')
    ax1.set_ylabel('Energy /$\mathrm{cm^{-1}}$')
    plt.plot(x1, y1, color="#006198")
    plt.show()

def main():
    print("vibrational frequency/cm^-1: ", hbaromega, '\n')
    H = Hamiltonian()   #Hamiltonian行列の生成
    print("dimension: ", len(H), '\n')
    value: np.ndarray = np.zeros(len(H))
    vector: np.ndarray = np.zeros((len(H), len(H)))
    value, vector = diagonalization(H)  #Hamiltonian行列の対角化
    print('eigen value\n{}\n'.format(value))
    MakeGraph_Potential(value, vector)  #option: ポテンシャルグラフの作成
    return

main()