import numericalsovler_eig as num_wf
import analytical_solution
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def animating_dirichlet(dt = 0.0005, t0=0, ts =0.2, L=1, mx=101):
    wf = num_wf.wavefunction(
        bcs=num_wf.boundary_conditions(
            left_BC  = num_wf.BoundaryCondition('dirichlet', tau=-1000),
            right_BC = num_wf.BoundaryCondition('dirichlet', tau=-1000),
        ), xr=L, x0=L/2, mx=mx, order=6
    )

    fig, ax = plt.subplots()

    for t in np.arange(t0,ts,dt):
        ax.clear()
        wf.time_evolution(t=t)
        x,y = wf.plot_probability()
        ax.plot(x,y, label='Numerisk')
        ax.plot(wf.gridpoints, np.abs(analytical_solution.analytical_solution_dirichlet(x0=L/2, xr=L, t=t, mx=mx))**2, label='Analytisk')
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
    plt.close()


def animating_neumann(dt = 0.0005, t0=0, ts =0.2, L=1, mx=101):
    wf = num_wf.wavefunction(
        num_wf.boundary_conditions(
            left_BC  = num_wf.BoundaryCondition('neumann', tau=-1000),
            right_BC = num_wf.BoundaryCondition('neumann', tau=-1000),
        ), xr=L, x0=L/2,mx=mx
    )
    
    fig, ax = plt.subplots()

    for t in np.arange(t0,ts,dt):
        ax.clear()
        wf.time_evolution(t=t)
        x,y = wf.plot_probability()
        ax.plot(x,y, label='Numerisk')
        ax.plot(wf.gridpoints, np.abs(analytical_solution.analytical_solution_nuemann(x0=L/2, xr=L, t=t,mx=mx))**2, label='Analytisk')
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        
        display.display(plt.gcf())
        display.clear_output(wait=True)

    plt.close()


def animating_radiation(dt = 0.0005, t0=0, ts =0.2, L=1, mx=101):
    wf = num_wf.wavefunction(
        bcs=num_wf.boundary_conditions(
            left_BC  = num_wf.BoundaryCondition('radiation', tau=-1),
            right_BC = num_wf.BoundaryCondition('radiation', tau=1),
        ), xr=L, x0=L/2, mx=mx, order=6
    )
    wf.time_evolution(t=t0)
    x, y = wf.plot_probability()
    y_max = np.max(y)

    fig, ax = plt.subplots()

    for t in np.arange(t0,ts,dt):
        ax.clear()
        wf.time_evolution(t=t)
        x,y = wf.plot_probability()
        ax.plot(x,y, label='Numerisk')
        # ax.plot(wf.gridpoints, np.abs(analytical_solution.analytical_solution_dirichlet(x0=L/2, xr=L, t=t, mx=mx))**2, label='Analytisk')
        ax.set_ylim(0, y_max * 1.1) 
        ax.legend()
        ax.set_title(f't = {t:.4f},  prob = {wf.total_prob():.4f}')
        
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
    plt.close()

# ---------------KÖR INTE FLERA SAMTIDIGT----------------- #

#animating_dirichlet()
#animating_neumann()
#animating_radiation()