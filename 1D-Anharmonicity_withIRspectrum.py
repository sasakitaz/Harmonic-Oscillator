"""
Hamiltonian
H = p^2/2mu + kxx x^2 + kxxx x^3

ex. HClの分子振動
omega = 2989 cm-1 ref. Herzberg
非調和項Vzzzは参考資料が見つからなかったため適当．
"""
import numpy as np
import matplotlib.pyplot as plt

#グローバル変数の定義
#matrix size (n0 + 1)
n0: int = 10

#molecular paramter (amu)
m_1: float = 1  #H atom
m_2: float = 35 #Cl atom
mu: float = m_1*m_2/(m_1 + m_2)

#potential paramter
kzz: float = 3000    #cm-1 A^-2
kzzz: float = -800   #cm-1 AA^-3

#constant
c: float = 2.99792458*10**8 #speed of light
hbar: float = 1.05457182*10**(-34)    #Planck constant
NA:float = 6.02214*10**(23)  #Avogadro constant
mu = mu/NA/10**3    #kg
kzz_si: float = kzz*1.98645*10**(-23)*10**(20)    #J m-2
hbaromega: float = np.sqrt(2*kzz_si/mu)/c/100     #cm-1

#calculation of matrixelement
class MatrixElement:
    def __init__(self, n0row, n0column):
        self.n0row: int = n0row
        self.n0column: int = n0column
        
    #ここで定義しているがこのプログラムでは使わない．
    def Creation(self):
        create: float = 0
        if self.n0row - self.n0column == + 1:
            create = np.sqrt(self.n0column + 1)
        else:
            create = 0
        return create
    
    def Annihilation(self):
        annihilate: float = 0
        if self.n0row - self.n0column == - 1:
            annihilate = np.sqrt(self.n0column)
        else:
            annihilate = 0
        return annihilate
    
    def T1(self):
        T1: float = 0
        if self.n0row - self.n0column == 0:
            T1 = 2*self.n0column + 1
        elif self.n0row - self.n0column == + 2:
            T1 = - np.sqrt((self.n0column + 1)*(self.n0column + 2))
        elif self.n0row - self.n0column == - 2:
            T1 = - np.sqrt(self.n0column*(self.n0column - 1))
        else:
            T1 = 0
        return T1
    
    def Vxx(self):
        Vxx: float = 0
        if self.n0row - self.n0column == 0:
            Vxx = 2*self.n0column + 1
        elif self.n0row - self.n0column == + 2:
            Vxx = np.sqrt((self.n0column + 1)*(self.n0column + 2))
        elif self.n0row - self.n0column == - 2:
            Vxx = np.sqrt(self.n0column*(self.n0column - 1))
        else:
            Vxx = 0
        return Vxx
    
    def Vxxx(self):
        Vxxx: float = 0
        if self.n0row - self.n0column == + 3:
            Vxxx = np.sqrt(self.n0column + 3)*np.sqrt(self.n0column + 2)*np.sqrt(self.n0column + 1)
        elif self.n0row - self.n0column == - 3:
            Vxxx = np.sqrt(self.n0column - 2)*np.sqrt(self.n0column - 1)*np.sqrt(self.n0column)
        elif self.n0row - self.n0column == + 1:
            Vxxx = 3*(self.n0column)*np.sqrt(self.n0column + 1) + 3*np.sqrt(self.n0column + 1)
        elif self.n0row - self.n0column == - 1:
            Vxxx = 3*(self.n0column)*np.sqrt(self.n0column - 1) + 3*np.sqrt(self.n0column)
        else:
            Vxxx = 0
        return Vxxx
    
#generation of Hamiltonian matrix
def Hamiltonian():
    result: np.ndarray = np.array([[+ MatrixElement(n0row, n0column).T1() *hbaromega/4
                                    + MatrixElement(n0row, n0column).Vxx() *hbaromega/4
                                    + MatrixElement(n0row, n0column).Vxxx() *(hbaromega/(4*kzz))**(3/2) *kzzz
            for n0column in range(0, n0 + 1)
        ]
        for n0row in range(0, n0 + 1)
    ])
    result = result.astype(np.float64)
    return result

#diagonarization and generation of eigen vector list
#eig_vec: raw data, N*N eigen vector matrix
#eigen_vector: the list sorted by quantum number 
def diagonalization(H):
    dim: int = len(H)
    eig_val: np.ndarray = np.zeros(dim)
    eig_vec: np.ndarray = np.zeros((dim, dim))
    eig_val,eig_vec = np.linalg.eigh(H)

    eigen_vector: list = []
    for r in range(0, len(eig_vec)):
        for c in range(0, len(eig_vec[r])):
            quantn0: int = c % dim
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantn0])
    return eig_val, eigen_vector

#option
def IR_intensity(eigen_vector):
    dim: int = int(np.sqrt(len(eigen_vector))) 
    eigen_vector: np.ndarray = np.array(eigen_vector)

    coeffs: list = []
    initstate: list = []
    for r in range(0, dim):
        num: list = eigen_vector[r*dim:(r + 1)*dim, :]
        if r == 0:
            initstate: list = num
        coeffs.append(num)
    
    #calculation of transition intensity
    intensity: list = []
    for i in range(0, dim):
        cfcg: float = 0
        for c in range (0, len(coeffs[i])):
            for cc in range (0, len(coeffs[i])):
                cg: float = initstate[cc][1]
                initn0: float = initstate[cc][2]
                cf: float = coeffs[i][c][1]
                n0:float = coeffs[i][c][2]
                if n0 - initn0 == + 1:
                    cfcg += cf*cg*np.sqrt(initn0 + 1)
                elif n0 - initn0 == - 1:
                    cfcg += cf*cg*np.sqrt(initn0)
                else:
                    cfcg += 0
        intensity.append([coeffs[i][0][0] - initstate[0][0], abs(cfcg)**2]) #transition energy, intensity
    return intensity

#option
def MakeGraph_IRspectrum(intensity):
    fig = plt.figure(dpi=300, figsize=(4,3))
    ax = fig.add_subplot(111)
    start = 0
    end = 10000
    grid = 101
    transition = [0]*grid
    wn = np.linspace(start, end, grid)
    for i in range (0, len(intensity)):
        for j in range (0, grid):
            if round(wn[j], -2) == round(intensity[i][0], -2):
                transition[j] += intensity[i][1]  
    fig.suptitle("IR spectrum")
    ax.text(0.7, 0.85, "$k_{zz}$ = " + str(kzz) + "\n$k_{zzz}$ = " + str(kzzz), fontsize = "11", transform = ax.transAxes)
    ax.bar(wn, transition, align="edge", width=100, color="crimson")
    ax.set_xlabel('Wavenumber/$\mathrm{cm^{-1}}$')
    ax.set_ylabel('Intensity (arb. unit)')
    plt.show()
    return

#option
def MakeGraph_Potential(eig_val):
    fig = plt.figure(dpi=300, figsize=(4,3))
    ax1 = fig.add_subplot(111)
    
    #Function
    x1 = np.linspace(-2.0, 2.0, 101)
    y1 = float(kzz)*x1**2 + float(kzzz)*x1**3
    
    #Eigen state
    for i in range (0, 3):  #見たい固有値の数だけ入力
        plt.hlines(eig_val[i], -1.5, 1.5, color="k")
    fig.suptitle("Potential energy surface of 1D oscillator")
    ax1.text(0.7, 0.85, "$k_{zz}$ = " + str(kzz) + "\n$k_{zzz}$ = " + str(kzzz), fontsize = "11", transform = ax1.transAxes)
    ax1.plot(x1, y1)
    ax1.set_xlabel('$\Delta$ Length/ $\mathrm{\AA}$')
    ax1.set_ylabel('Energy /$\mathrm{cm^{-1}}$')
    plt.plot(x1, y1, color="#006198")
    plt.show()
    return

def main():
    print('matrix size: ', n0 , '\n')
    print("vibrational frequency/cm^-1: ", hbaromega, '\n')
    H: np.ndarray = Hamiltonian()   #Hamiltonian行列の生成
    value: np.ndarray = np.zeros(len(H))
    vector: np.ndarray = np.zeros((len(H), len(H)))
    value, vector = diagonalization(H)  #Hamiltonian行列の対角化
    print('eigen value\n{}\n'.format(value))
    MakeGraph_Potential(value) #option: ポテンシャルグラフの作成
    MakeGraph_IRspectrum(IR_intensity(vector)) #option: IRスペクトルグラフの作成
    return

main()