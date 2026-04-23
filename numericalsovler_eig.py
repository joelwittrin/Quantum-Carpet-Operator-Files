import numpy as np
import matplotlib.pyplot as plt
import operators as ops
from scipy.integrate import simps


def Gaussian_IC(x, x0=0.5, l0=0.1,k0=0):
    #Regular Gaussian Profile

    psi = np.exp(-((x - x0) / l0)**2)
    A0 = 1 / np.sqrt(simps(np.abs(psi)**2, x))
    return (A0 * psi).astype(complex)


def Gaussian_IC_v2(x, x0=0.5, l0=0.2,k0=1j):
    #Gaussian Profile with Complex exponent, specifically with Sommerfeld Condition in mind

    psi = np.exp(-((x - x0) / l0)**2)*(np.exp(1j * k0 * (x-x0))+np.exp(-1j *k0 * (x-x0)))
    A0  = 1 / np.sqrt(np.trapz(np.abs(psi)**2, x))
    return (A0 * psi).astype(complex)


def zero_potential(x):
    return x*0


class BoundaryCondition:
    #Middle step in order to change BC computations without interfering with the wave class itself

    def __init__(self, bc_type, tau,k0=1):
        self.bc_type = bc_type
        self.tau     = tau
        self.k0      = k0


class boundary_conditions:
    #Discretization of chosen BC: Dirichlet, Neumann or Radiation

    def __init__(self, left_BC, right_BC):
        self.left_BC  = left_BC
        self.right_BC = right_BC


    def compute_sat_v2(self, HI, e_l, e_r, d1_l, d1_r):
        #Returns SAT-terms before multiplying with the state Psi
        
        # Left BC
        if self.left_BC.bc_type == 'dirichlet':
            sat_left = self.left_BC.tau * HI @ e_l.T @ e_l
            
        elif self.left_BC.bc_type == 'neumann':

            sat_left = self.left_BC.tau * HI @ e_l.T @ d1_l
        
        elif self.left_BC.bc_type == 'radiation':
            sat_left = self.left_BC.tau * HI @ e_l.T@(d1_l+1j*self.left_BC.k0*e_l)
        else:
            sat_left = 'error'

        # Right BC
        if self.right_BC.bc_type == 'dirichlet':
            sat_right = self.right_BC.tau * HI @ e_r.T @ e_r
            
        elif self.right_BC.bc_type == 'neumann':
            sat_right = self.right_BC.tau * HI @ e_r.T @ d1_r

        elif self.right_BC.bc_type == 'radiation':
            sat_right = self.right_BC.tau * HI @ e_r.T@(d1_r-1j*self.left_BC.k0*e_r)

        else:
            sat_right = 'error'

        return sat_left + sat_right


class wavefunction:
    #All computation needed for handling the wave function from a given IC

    def __init__(
        self,
        bcs = boundary_conditions(
            left_BC=BoundaryCondition('dirichlet', tau=-50),
            right_BC=BoundaryCondition('dirichlet', tau=-50),
        ),
        v=zero_potential,
        xl=0,
        xr=1,
        order=2,
        IC=Gaussian_IC,
        x0=0.5,
        mx=101,
        dt = 0.00001,
        k0=1j,
                ):

        #Initialize grid step hx
        self.hx = (xr - xl) / (mx - 1)

        if order == 2:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_2nd(mx, self.hx)
        elif order == 4:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_4th(mx, self.hx)
        elif order == 6:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_6th(mx, self.hx)

        self.bcs = bcs
        self.time = 0
        self.dt = dt
        self.gridpoints = np.linspace(xl, xr, mx)

        #V has to be a matrix mx*mx with the discrete potential on the diagonal and the rest elements are zero...
        #so that Psi can be factored out and D2 and V can be added together
        self.V = np.diag(v(self.gridpoints))

        #The initial condition sampled at the gridpoints
        self.state = IC(self.gridpoints, x0=x0,k0=k0)

        # Attributes that are given values in solve_eigenvalue_problem
        self.cn0 = None
        self.cn = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.solve_eigenvalue_problem()

    def rhs_matrix(self):
        #Defines M when we have the independent Schrödinger eq. on the form Eϕ = Mϕ...
        #where M is the combined matricies on the right hand side
        sat_terms = self.bcs.compute_sat_v2(self.HI, self.e_l, self.e_r, self.d1_l, self.d1_r)
        
        '''Unclear if we are supposed to have +D2 or -D2'''

        return -self.D2  +self.V  + sat_terms

    def solve_eigenvalue_problem(self):
        # Eigenvalues to matrix M correpsond to the energies!

        # Eϕ = -D2ϕ + Vhϕ + HI*tau_l*e_l(e_l.Tϕ) + HI*tau_r*e_r(e_r.Tϕ)
        # Eϕ = (-D2 + Vh + HI*tau_l*e_l@e_l.T + HI*tau_r*e_r@e_r.T) ϕ
        # M = -self.D2 +self.V +self.tau_l*self.HI*self.e_l.T@self.e_l +self.tau_r*self.HI*self.e_r.T@self.e_r
        M = self.rhs_matrix()
        self.eigenvalues, self.eigenvectors = np.linalg.eig(M)
        self.cn0 = np.asarray(self.eigenvectors.T @ self.state).flatten()
        self.cn = self.cn0

        #debug:
        #print(np.linalg.norm(self.HI@M-self.HI@M.transpose(),'fro')/np.linalg.norm(self.HI@M,'fro'))
        return None
    

    def plot_probability(self):
        return self.gridpoints, np.abs(np.asarray(self.state).flatten())**2

    def time_evolution(self, t, hbar=1):
        #Since all Eigenvalues are computed in solve_eigenvalue_problem() any time step can be plotted...
        #i.e. no need for iterative method to find specific value att time t

        self.cn=self.cn0*np.exp(-1j*self.eigenvalues*t/hbar)
        self.state = (self.eigenvectors @ self.cn)
        return None

    def total_prob(self):
        #Probability, when simulating this should always return ~1

        return np.trapz(np.abs(np.asarray(self.state).flatten())**2, self.gridpoints)
    
    def quantum_carpet_matrix(self, ts):
        #Build the Quantum Carpet, rows=timestep & columns = gridpoints

        QCM = [np.abs(self.state)**2]
        while self.time <= ts:
            self.time += self.dt
            self.time_evolution(t=self.time) 
            QCM.append(np.abs(np.asarray(self.state).flatten())**2)
        QCM = np.array(QCM)
        return QCM
    

class wavefunction_rad:
    def __init__(
        self,
        tau_r = 1,
        tau_l = -1,
        v=zero_potential,
        xl=0,
        xr=1,
        order=2,
        IC=Gaussian_IC,
        x0=0.5,
        mx=101,
        dt = 0.00001,
    ):
        self.hx = (xr - xl) / (mx - 1)

        if order == 2:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_2nd(mx, self.hx)
        elif order == 4:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_4th(mx, self.hx)
        elif order == 6:
            self.H, self.HI, self.D1, self.D2, self.e_l, self.e_r, self.d1_l, self.d1_r = ops.sbp_cent_6th(mx, self.hx)

        self.tau_r = tau_r
        self.tau_l = tau_l
        self.time = 0
        self.dt = dt
        self.gridpoints = np.linspace(xl, xr, mx)
        ### V has to be a matrix mx*mx with the discrete potential on the diagonal and the rest elements are zero
        ### So that Psi can be factored out... And D2 and V can be added together
        self.V          = np.diag(v(self.gridpoints))
        #The initial condition sampled at the gridpoints
        self.state      = IC(self.gridpoints, x0=x0)
        ### Attributes that are given values in solve_eigenvalue_problem
        self.cn0 = None
        self.cn = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.solve_eigenvalue_problem()
    

    def solve_eigenvalue_problem(self):
        ### CM = constant matrix ie no energy dependent in it
        ### includes the sat terms which does not contain sqrt(E) if we multiply the parenthesis
        CM = (
            -self.D2  
            +self.V 
            +self.tau_l * self.HI @ self.e_l.T@self.d1_l
            +self.tau_r * self.HI @ self.e_r.T@self.d1_r
        )
        ### sqEM square_root_E_matrix is the matrix from the sat terms that include sqrt(E)
        ### but with sqrt(E) factored out
        sqEM = (
            self.tau_l * self.HI @ self.e_l.T@1j*self.e_l
            + self.tau_r * self.HI @ self.e_r.T@(-1j*self.e_r)
        )

        ### we now have an eigenvalue problem on the form: 
        ### (CM +sqEM*sqrt(E))*PSI = E*PSI
        ### let k = sqrt(E)
        ### (CM+sqEM*k)*Psi = k**2*psi
        ### (k**2*I-k*sqEM-CM)psi = 0


        return 
    
    def plot_probability(self):
        return self.gridpoints, np.abs(np.asarray(self.state).flatten())**2

    def time_evolution(self, t, hbar=1):
        self.cn=self.cn0*np.exp(-1j*self.eigenvalues*t/hbar)
        self.state = (self.eigenvectors @ self.cn)
        return None

    def total_prob(self):
        return np.trapz(np.abs(np.asarray(self.state).flatten())**2, self.gridpoints)
    
    def quantum_carpet_matrix(self, ts):
        QCM = [np.abs(self.state)**2]
        while self.time <= ts:
            self.time += self.dt
            self.time_evolution(t=self.time) 
            QCM.append(np.abs(np.asarray(self.state).flatten())**2)
        QCM = np.array(QCM)
        return QCM
        