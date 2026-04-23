import numpy as np
import matplotlib.pyplot as plt
import operators as ops
# import Convergence

def analytical_solution_dirichlet(l0=0.1,t=0,x0=0.5,xl=0, xr=1, mx=101,n_terms=50, hbar=1.0, m=0.5):
  gridpoints = np.linspace(xl,xr,mx)
  L=xr-xl
  gridpoints = np.asarray(gridpoints, dtype=complex)
  prefactor = 2/L*(2 * np.pi * l0**2) ** 0.25

  total = np.zeros_like(gridpoints, dtype=complex)

  for n in range(1, n_terms + 1):
      k = n * np.pi / L
      decay = np.exp(-k**2 * (l0**2 / 4 + 1j * hbar * t / (2 * m)))
      total += decay * np.sin(k * x0) * np.sin(k * gridpoints)

  return prefactor * total

def analytical_solution_nuemann(l0=0.1,t=0,x0=0.5,xl=0, xr=1, mx=101,n_terms=50, hbar=1.0, m=0.5):
  gridpoints = np.linspace(xl,xr,mx)
  L=xr-xl
  gridpoints = np.asarray(gridpoints, dtype=complex)
  prefactor = 2/L*(2 * np.pi * l0**2) ** 0.25

  total = np.zeros_like(gridpoints, dtype=complex)

  #n=0 term
  total += 1/2

  for n in range(1, n_terms + 1):
      k = n * np.pi / L
      decay = np.exp(-k**2 * (l0**2 / 4 + 1j * hbar * t / (2 * m)))
      total += decay * np.cos(k * x0) * np.cos(k * gridpoints)

  return prefactor * total


def L2_error(analytical_state_1, analytical_state_2,hx):    
  return np.sqrt(np.sum(np.square(np.abs(analytical_state_2-analytical_state_1)*hx)))

def test_sufficient_N(N=40, xl=0, xr=1, mx=101,):
  hx = (xr-xl)/(mx-1)
  analytical_state_1 = analytical_solution_nuemann(l0=0.1,t=0,x0=0.5,xl=xl, xr=xr, mx=mx,n_terms=N, hbar=1.0, m=0.5)
  analytical_state_2 = analytical_solution_nuemann(l0=0.1,t=0,x0=0.5,xl=xl, xr=xr, mx=mx,n_terms=2*N, hbar=1.0, m=0.5)
  return print(L2_error(analytical_state_1, analytical_state_2,hx))
   

# test_sufficient_N()
   
