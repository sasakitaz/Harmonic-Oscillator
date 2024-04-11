"""
Hamiltonian
H = p^2/2mu + kxx (x^2 + y^2)

動径方向に非調和性を持つポテンシャル
"""
import numpy as np
import matplotlib.pyplot as plt

#グローバル変数の定義
#matrix size
n2: int = 2
npls: int = n2
nmns: int = n2

#molecular paramter (amu)
mu: float = 1

#potential paramter
kxx: float = 2000    #cm-1 A^-2

#constant
c = 2.998*10**8 #speed of light
hbar = 1.05457*10**(-34)    #Planck constant
NA = 6.02*10**(23)  #Avogadro constant
mu = mu/NA/10**3    #kg
kxx_si = kxx*1.98645*10**(-23)*10**(20)    #J m-2
hbaromega = np.sqrt(2*kxx_si/mu)/c/100     #cm-1

#calculation of matrixelement
class MatrixElement:
    def __init__(self, nplsrow, nmnsrow, nplscolumn, nmnscolumn):
        self.nplsrow: int = nplsrow
        self.nmnsrow: int = nmnsrow
        self.vrow: int = nplsrow + nmnsrow
        self.lrow: int = nplsrow - nmnsrow
        self.nplscolumn: int = nplscolumn
        self.nmnscolumn: int = nmnscolumn
        self.vcolumn: int = nplscolumn + nmnscolumn
        self.lcolumn: int = nplscolumn - nmnscolumn
    
    def T_2D(self):  
        me_T_2D: float = 0
        if self.nplsrow == self.nplscolumn + 1 and self.nmnsrow == self.nmnscolumn + 1:
            me_T_2D = - 2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))
        elif  self.nplsrow == self.nplscolumn - 1 and self.nmnsrow == self.nmnscolumn - 1:
            me_T_2D = - 2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))
        elif  self.nplsrow == self.nplscolumn and self.nmnsrow == self.nmnscolumn:
            me_T_2D = 2*(self.nplscolumn + self.nmnscolumn + 1)
        else:
            me_T_2D = 0
        return me_T_2D
    
    def Vxx(self):
        me_Vxx: float = 0
        if self.nplsrow - self.nplscolumn == + 1 and self.nmnsrow - self.nmnscolumn == + 1:
            me_Vxx = 2*np.sqrt((self.nplscolumn + 1)*(self.nmnscolumn + 1))                  
        elif  self.nplsrow - self.nplscolumn == - 1 and self.nmnsrow - self.nmnscolumn == - 1:
            me_Vxx = 2*np.sqrt((self.nplscolumn)*(self.nmnscolumn))
        elif  self.nplsrow - self.nplscolumn == 0 and self.nmnsrow - self.nmnscolumn == 0:
            me_Vxx = 2*(self.nplscolumn + self.nmnscolumn + 1)
        else:
            me_Vxx = 0          
        return me_Vxx

def Hamiltonian():
    result: np.ndarray = np.array([[+ MatrixElement(nplsrow, nmnsrow, nplscolumn, nmnscolumn).T_2D() *hbaromega/4
                                    + MatrixElement(nplsrow, nmnsrow, nplscolumn, nmnscolumn).Vxx() *hbaromega/4
                for nplscolumn in range(0, npls + 1)
                for nmnscolumn in range(0, nmns + 1)    
            ]
            for nplsrow in range(0, npls + 1)
            for nmnsrow in range(0, nmns + 1)    
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
    for r in range(0, len(eig_vec)):
        for c in range(0, len(eig_vec[r])):
            quantnpls: int = c // (npls + 1)
            quantnmns: int = c % (nmns + 1)
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantnpls, quantnmns])
    return eig_val, eigen_vector

#option
def MakeGraph_Potential(eig_val, eig_vec):
    fig = plt.figure(dpi=300, figsize=(4,3))
    ax1 = fig.add_subplot(111)
    
    #Function
    x1 = np.linspace(0, 2.0, 101)
    y1 = float(kxx)*x1**2
    
    for ii in range (0, len(eig_val)):
        plt.hlines(eig_val[ii], -0.5, -0.1, color="k")
    
    #Eigen state
    for ii in range (0, len(eig_vec)):
        if abs(eig_vec[ii][1]) > 0.5:
            if abs(eig_vec[ii][2] - eig_vec[ii][3]) == 0:
                plt.hlines(eig_vec[ii][0], 0, 0.4, color="k")
            elif abs(eig_vec[ii][2] - eig_vec[ii][3]) == 1:
                plt.hlines(eig_vec[ii][0], 0.5, 0.9, color="k")
            elif abs(eig_vec[ii][2] - eig_vec[ii][3]) == 2:
                plt.hlines(eig_vec[ii][0], 1.0, 1.4, color="k")
            elif abs(eig_vec[ii][2]) - eig_vec[ii][3] == 3:
                plt.hlines(eig_vec[ii][0], 1.5, 1.9, color="k")
    ax1.text(0.7, 0.85, "$k_{zz}$ = " + str(kxx), fontsize = "11", transform = ax1.transAxes)
    ax1.text(-0.5, 0, "all")
    ax1.text( 0.0, 0, "$l = 0$(s)")
    ax1.text( 0.5, 0, "$l = 1$(p)")
    ax1.text( 1.0, 0, "$l = 2$(d)")
    ax1.text( 1.5, 0, "$l = 3$(f)")
    fig.suptitle("Radial potential of 2D oscillator")
    ax1.plot(x1, y1)
    ax1.set_ylim(-500, 10000)
    ax1.set_xlabel('radial/ $\mathrm{\AA}$')
    ax1.set_ylabel('Energy /$\mathrm{cm^{-1}}$')
    plt.plot(x1, y1, color="#006198")
    plt.show()

def main():
    print('matrix size: ', (npls + 1)*(nmns + 1) , '\n')
    print("vibrational frequency/cm^-1: ", hbaromega, '\n')
    value: np.ndarray = np.zeros(len(H))
    vector: np.ndarray = np.zeros((len(H), len(H)))
    H = Hamiltonian()   #Hamiltonian行列の生成
    value, vector = diagonalization(H)  #Hamiltonian行列の対角化
    print('eigen value\n{}\n'.format(value))
    MakeGraph_Potential(value, vector)  #option: ポテンシャルグラフの作成
    return

main()