# -*- coding: utf-8 -*-
r"""
Last updated: Wed Mar 18 11:09:02 2026

Uploaded GitHub Version: 
    - Calculates the optimal theta values by optimizing beta, fid, or
      trace norm distance
    - Plots the CHSH S function for various N
    - Outputs data file: beta, S, fidelity, trace norm, final energy
      ASCII art: -ANSI-paga https://www.asciiart.eu/text-to-ascii-art
                             
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ░█▀▄░█▀▀░█░░░█░░░░░▀█▀░█▀▀░█▀▀░▀█▀░░░█▀▀░█▀█░█▀▄░░░█▄█░█▀▀░█░█            
# ░█▀▄░█▀▀░█░░░█░░░░░░█░░█▀▀░▀▀█░░█░░░░█▀▀░█░█░█▀▄░░░█░█░█▀▀░▀▄▀            
# ░▀▀░░▀▀▀░▀▀▀░▀▀▀░░░░▀░░▀▀▀░▀▀▀░░▀░░░░▀░░░▀▀▀░▀░▀░░░▀░▀░▀▀▀░░▀░            
# ░█▀█░█░█░█▀█░▀█▀░█▀█░█▀█░█▀▀                                              
# ░█▀▀░█▀█░█░█░░█░░█░█░█░█░▀▀█                                              
# ░▀░░░▀░▀░▀▀▀░░▀░░▀▀▀░▀░▀░▀▀▀  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ░█▀█░█▀█░█░█░█▄█░░░░░█▀▄░█▀█░█▀▀░█▀▀░█▀▄░░░█▀█░█▀█░█▀█░█░░░█░█░█▀▀░▀█▀░█▀▀
# ░█▀▀░█░█░▀▄▀░█░█░▄▄▄░█▀▄░█▀█░▀▀█░█▀▀░█░█░░░█▀█░█░█░█▀█░█░░░░█░░▀▀█░░█░░▀▀█
# ░▀░░░▀▀▀░░▀░░▀░▀░░░░░▀▀░░▀░▀░▀▀▀░▀▀▀░▀▀░░░░▀░▀░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀▀▀
# ░█▀█░█▀▀░░░█▀▀░█▀▀░▄▀▄░█░█░█▀▀░█▀█░▀█▀░▀█▀░█▀█░█░░                        
# ░█░█░█▀▀░░░▀▀█░█▀▀░█\█░█░█░█▀▀░█░█░░█░░░█░░█▀█░█░░                        
# ░▀▀▀░▀░░░░░▀▀▀░▀▀▀░░▀\░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀                        
# ░█▀▀░█▀█░█▄█░█▀█░▀█▀░█▀█░█▀█░░░█▀▀░█▀▀░█▀█░▀█▀░▀█▀░█▀▀░█▀▄░▀█▀░█▀█░█▀▀    
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░░░▀▀█░█░░░█▀█░░█░░░█░░█▀▀░█▀▄░░█░░█░█░█░█    
# ░▀▀▀░▀▀▀░▀░▀░▀░░░░▀░░▀▀▀░▀░▀░░░▀▀▀░▀▀▀░▀░▀░░▀░░░▀░░▀▀▀░▀░▀░▀▀▀░▀░▀░▀▀▀                                     
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ________________________________________________________________________________________________
    - Finds the angles {theta1,theta2,...,thetan} to maximize fidelity 
      (or minimize trace Norm) of POVM with the horizontal projector.
    - Uses the optimal angles to find the maximal violation in CHSH inequality. 
    - Plots the S(phi) function. 
    - E0=1 is set for para-Ps but can be varied if needed.
                                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                    

@author: Jack Clarke 
         University College London
         
         Citation for the manuscript: 
         Jack Clarke, Preslav Asenov, Jesse Smeets, Jia-Shian Wang, 
         David B. Cassidy, and Alessio Serafini, 
         "Bell Test for MeV Photons via POVM-based Compton Polarimetry",
         arXiv:2604.25034 (2026)
"""
#%%

#~Import Packages~#
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl 


r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ░█▀▀░█▀▀░▀█▀░█░█░█▀█
    # ░▀▀█░█▀▀░░█░░█░█░█▀▀
    # ░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░░
                             
    - Stokes Vector for a General Qubit in Circularly Polarized Basis
      (Right and Left correspond to North and South poles of Bloch sphere)
    - Horizontal Eigenstate
    - Fidelity and Trace Norm Functions
    - Bell state from para-positronium in Circularly Polarized Basis
    - Probability
    - V5 NEW SIMPLIFIED FUNCTIONS THAT JUST DEPEND ON BETA
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#~Stokes Vector Basis~#
" With np.shape(Sij)=(4,1), i.e. column vectors. "
SRR=np.array([[1,0,0,1]]).T
SLL=np.array([[1,0,0,-1]]).T
SRL=np.array([[0,1,1j,0]]).T
SLR=np.array([[0,1,-1j,0]]).T

#~V6: Unpolarized Stokes vector~#
Sunpol=np.array([[1,0,0,0]]).T

#~Horizontal Eigenstate~#
"Eigenstate along +x direction of Bloch Sphere"
H=(1/np.sqrt(2))*np.array([[1,1]]).T

#~Rotation Matrix along Equator of Bloch Sphere~#
"""Pauli-z generates rotation between H->D->V->A
   Clockwise round Bloch Sphere
   Factor of 2 difference to standard definition is to match defintion
   of Mphi where phi is a physical angle.
"""
def Rphi(phi):
    return np.array([[np.exp(1j*phi),0],[0,np.exp(-1j*phi)]])

#~Intensity Measurment on Stokes Vector~#
ID=np.array([[1,0,0,0]]).T

#~Bell State from para-Ps Decay~#
rhoBell=0.5*np.array([[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]])

#~Fidelity Function~#
" A is a 2x2 POVM and v is a 2D column vector projector. "
def Fid(A,v):
    vAv=(v.conj().T) @ A @v 
    return np.real(vAv[0,0])

#~Fidelity Function SIMPLIFIED~#
" Fidelity only depends on beta parameter. beta is real"
def FidB(beta):
    return 0.5*(1+np.real(beta))

#~Trace Norm Function~#
""" A is a 2x2 POVM and v is a 2D column vector projector. 
    Note this function is defined slightly differently to 
    Mathematica Notebook.
    Furthermore, I include +epsilon*Id, so no singular value can ever be zero.
    Otherwise, the multivariable solver below returns:
        Optimization failed: Singular matrix E in LSQ subproblem
"""
def Td(A,v):
    epsilon = 1e-12 * np.max(np.abs(A)) # small epislon for stablility
    Diff=A-np.outer(v,v.conj())
    # Add a small scalar multiple of the identity matrix for stability
    stabilized_Diff = Diff + epsilon * np.eye(Diff.shape[0])
    """ Set full_matrices=False, so that Autograd is able to compute the 
        Jacobian.
    """
    U, S, Vt = np.linalg.svd(stabilized_Diff, full_matrices=False)
    Tnorm=0.5*np.sum(S)
    return Tnorm

#~Trace Norm Function SIMPLIFIED~#
" Trace Norm only depends on beta parameter. Beta is real."
def TdB(beta):
    return 0.5*(1-np.real(beta))

#~CHSH Function SIMPLIFIED~#
" CHSH FUNCTION only depends on beta parameter and phi. "
def SB(beta,phi):
    return np.real(beta)**2*np.abs(-3*np.cos(2*phi)+np.cos(6*phi))


#~Probability~#
"rho is a nxn density matrix, A is a nxn POVM"
def Pr(rho,A):
    return np.trace(rho @ A)   
    

r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ░█▀▀░█▀█░█▄█░█▀█░▀█▀░█▀█░█▀█░░░█▀▀░█▀▀░█▀█░▀█▀░▀█▀░█▀▀░█▀▄░▀█▀░█▀█░█▀▀
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░░░▀▀█░█░░░█▀█░░█░░░█░░█▀▀░█▀▄░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░░▀░░▀▀▀░▀░▀░░░▀▀▀░▀▀▀░▀░▀░░▀░░░▀░░▀▀▀░▀░▀░▀▀▀░▀░▀░▀▀▀
# ░█▄█░█▀█░▀█▀░█▀▄░▀█▀░█░█                                              
# ░█░█░█▀█░░█░░█▀▄░░█░░▄▀▄                                              
# ░▀░▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀   
    - Compton Scattering Matrix 
    - Rotation matrix M(phi)     
    - V5: NEW UPPER BLOCK (A) OF COMPTON SCATTERING MATRIX  
                                                 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#~Output Energy~#
def Eout(E0,theta):
    return E0/(1+E0*(1-np.cos(theta)))

#~MATRIX ELEMENTS~#
" t11 "
def t11(E0,theta):
    return 1 + np.cos(theta)**2 + (E0 - Eout(E0,theta))*(1 - np.cos(theta))
" t12 "
def t12(theta):
    return np.sin(theta)**2
" t33 "
def t33(theta):
    return 2*np.cos(theta)
" t44 "
def t44(E0,theta):
    return 2*np.cos(theta) + (E0 - Eout(E0, theta))*(1 - np.cos(theta))*np.cos(theta)

#~COMPTON SCATTERING MATRIX~#
"Note this is BLOCK diagonal"
def Tun(E0,theta):
    return 0.5*np.array([[t11(E0,theta), t12(theta), 0, 0],\
                         [t12(theta), 2 - t12(theta), 0, 0],\
                         [0, 0, t33(theta), 0],\
                         [0, 0, 0, t44(E0,theta)]])
        
#~UPPER BLOCK OF COMPTON SCATTERING MATRIX~#
def Aun(E0,theta):
    return 0.5*np.array([[t11(E0,theta), t12(theta)],\
                         [t12(theta), 2 - t12(theta)]])

        
#~Rotation Matrix~#
def Mphi(phi):
    return np.array([[1, 0, 0, 0], [0, np.cos(2*phi), np.sin(2*phi), 0],\
                     [0, -np.sin(2*phi), np.cos(2*phi), 0], [0, 0, 0, 1]])


r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ░█▀▀░█▀█░█▀▄░█▀▀░░░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
    # ░█░░░█░█░█▀▄░█▀▀░░░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
    # ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀
                             
    - V2: All defined for N scattering events
    - Differential Cross Section for Bell test trajectories
    - V5: NEW FUNCTION TO CALCUATE BETA
    - Unnormalized POVM
    - (Trace) Normalized POVM
    - V6: Differential Cross Section for General trajectories
    - V6: Properly normalized POVM
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#~Differential Cross Section~#
def sigma_N(theta_vec, phi, S):
    """
    Computes the differential cross-section (sigma) after N scattering events.
    The polarization rotation Mphi(phi) is applied to the initial Stokes vector S 
    *before* the first scattering event (Tun_1).
    
    The chain is: S_N = Tun_N @ ... @ Tun_1 @ [Mphi(phi) @ S]
    """
    
    # --- 1. Apply the Rotation Mphi(phi) to the Initial Stokes Vector ---
    S_current = Mphi(phi) @ S
    
    E_0=1.0           # Initial energy E0 for the first scattering (Tun_1)
    E_incident = 1.0  # Set to E0

    # --- 2. Iterate through the N Scattering Events ---
    # The loop applies the entire chain Tun_N @ ... @ Tun_1 to S_current
    for theta_n in theta_vec:
        # The matrix uses the energy *before* scattering
        Tun_n = Tun(E_incident, theta_n)
        
        # S_n = Tun_n @ S_{n-1}
        S_current = Tun_n @ S_current
        
        # Update the energy for the next step (E_{n} = Eout(E_{n-1}, theta_n))
        E_incident = Eout(E_incident, theta_n)
        
    E_final = E_incident  # The last calculated E_incident is the final energy E_N
    S_final = S_current
    
    # Final result: (E_N/E0)^2 * (ID^T @ S_N)
    return ((E_final/E_0)**2 * ((ID.T) @ S_final))[0,0]

#~v6: Differential Cross Section for general trajectories~#
def sigma_N_general(theta_vec, phi_vec, S_initial):
    """
    Computes the N-fold multi-differential cross section for a general trajectory.
    Trajectory: S_N = T(theta_N) @ M(phi_N) @ ... @ T(theta_1) @ M(phi_1) @ S_initial
    
    Args:
        theta_vec (list/array): [theta_1, theta_2, ..., theta_N]
        phi_vec (list/array):   [phi_1, phi_2, ..., phi_N]
        S_initial (array):      Initial 4x1 Stokes vector
    """
    S_current = S_initial
    E_0=1.0           # Initial energy E0 for the first scattering (Tun_1)
    E_incident = 1.0  # Set to E0
     

    # Apply the chain of (M_i then T_i)
    for i in range(len(theta_vec)):
        theta_i = theta_vec[i]
        phi_i = phi_vec[i]
        
        # 1. Rotate to the i-th scattering plane
        S_current = Mphi(phi_i) @ S_current
        
        # 2. Scatter at theta_i
        # The matrix Tun uses the energy *before* this specific scattering
        S_current = Tun(E_incident, theta_i) @ S_current
        
        # 3. Update energy for the next scattering event
        E_incident = Eout(E_incident, theta_i)
        
    E_final = E_incident
    
    # The differential cross section is the intensity component (index 0)
    # scaled by the final energy squared (E_N/E_0)^2
    return ((E_final/E_0)**2 * ((ID.T) @ S_current))[0, 0]

#~V5: CALCULATE THE BETA PARAMETER~#
def beta_N(theta_vec):
    """
    Computes the final upper block after N scattering events..
    
    The chain is: A_N @ ... @ A_1
    """
    
    E_incident = 1.0  # Initial energy E0 for the first scattering (Tun_1)
    
    A_current = np.array([[1, 0],[0, 1]]) # Initial A0 is identity
    
    # --- 1. Iterate through the N Scattering Events ---
    # The loop applies the entire chain A_N @ ... @ A_1
    for theta_n in theta_vec:
        # The matrix uses the energy *before* scattering
        A_n = Aun(E_incident, theta_n)
        
        # A_n = Tun_n @ S_{n-1}
        A_current = A_n @ A_current
        
        # Update the energy for the next step (E_{n} = Eout(E_{n-1}, theta_n))
        E_incident = Eout(E_incident, theta_n)
        
    #E_final = E_incident  # The last calculated E_incident is the final energy E_N
    A_final = A_current
    
    # The beta parameteter is given by the 1,2 element divided by the 1,1 element
    # Add a small safety net on the denominator, for incredily high N
    beta = A_final[0,1] / (A_final[0,0] + 1e-20)
    
    return beta


#~Unnormalized POVM~#
# Note factors of sigma_tot cancel in S function calculations for a given N.
# V7: FIXED THE ASSIGNMENT OF POVM ELEMENTS, i.e. rhoRL=1-->PiLR and vice versa
def Pi_N_Unnorm(theta_vec,phi):
    """
    Computes the UNNORMALIZED POVM Pi_N(theta_vec) using the sigma function.
    This will be useful for Bell tests and probabilities.
    """
    sigma_rr = sigma_N(theta_vec, phi, SRR)
    sigma_rl = sigma_N(theta_vec, phi, SLR)
    sigma_lr = sigma_N(theta_vec, phi, SRL)
    sigma_ll = sigma_N(theta_vec, phi, SLL)
    
    # The POVM is always a 2x2 matrix that acts on photon polarization subspace
    return np.array([[sigma_rr, sigma_rl],
                     [sigma_lr, sigma_ll]])

#~(Trace) Normalized POVM~#
def Pi_N_Norm(theta_vec, phi):
    """
    Computes the normalized POVM Pi_N_Norm(theta_vec) using the sigma function.
    This will be useful for fidelity (and trace norm difference) with |H><H|.
    """
    Pi_N=Pi_N_Unnorm(theta_vec,phi)
    
    # Normalize by the trace
    return Pi_N / np.trace(Pi_N)



r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
░█▀▀░█░░░█▀▀░▄▀▄░█▀█░░░█▀█░█▀█░▀█▀░▀█▀░█▄█░▀█▀░▀▀█░█▀█░▀█▀░▀█▀░█▀█░█▀█
░▀▀█░█░░░▀▀█░█\█░█▀▀░░░█░█░█▀▀░░█░░░█░░█░█░░█░░▄▀░░█▀█░░█░░░█░░█░█░█░█
░▀▀▀░▀▀▀░▀▀▀░░▀\░▀░░░░░▀▀▀░▀░░░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀
                             
    - Optimization Strategy: Sequential Least Squares Programming (SLQP) method.
    - Generalizes V1 to N scattering events
    - Optimizes with objective function = fidelity or trace norm
    - Generalizes V2 to include optional phi input
    - V5: NEW OPTIMIZATION STRATEGY TO JUST MAXIMIZE BETA
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#~Optimization~#
def run_optimization_N(N, objective='Fidelity', fixed_phi=0.0):
    """
    Sets up and runs the SLSQP optimization for N scattering events (thetas).
    Phi is treated as a fixed parameter.
    
    Args:
        N (int): The number of scattering events (angles to optimize).
        objective (str): 'Fidelity' or 'TraceNorm'.
        fixed_phi (float): The fixed value of the rotation angle phi.
    """
    
    print(f"\n--- Running Optimization for N = {N} Events ({objective}) with Fixed Phi = {fixed_phi:.4f} ---")

    # --- Setup Objective Function and Jacobian ---
    
    if objective == 'Fidelity':
        # Minimize -Fidelity. The inner function uses the fixed_phi value from the outer scope.
        def obj_fun(theta_vec):
            "Need to multiply H by phi here"
            return -Fid(Pi_N_Norm(theta_vec, fixed_phi), Rphi(fixed_phi) @ H) # <-- phi is passed here
        jac_fun = grad(obj_fun)
        
    elif objective == 'TraceNorm':
        # Minimize Trace Norm Difference
        def obj_fun(theta_vec):
            "Need to multiply H by phi here"
            return Td(Pi_N_Norm(theta_vec, fixed_phi), Rphi(fixed_phi) @ H) # <-- phi is passed here
        jac_fun = grad(obj_fun)
        
    else:
        raise ValueError("Objective must be 'Fidelity' or 'TraceNorm'")

    # --- Setup N-dimensional Inputs ---
    
    # Inputs are only the N theta angles
    bounds = [(0, np.pi)] * N
    np.random.seed(42) 
    theta0 = np.array([np.pi/4 + np.random.uniform(-0.5, 0.5) for _ in range(N)])

    print(f"Initial Guess (theta0): {theta0}")

    # --- Run the Minimization ---
    
    result = minimize(
        fun=obj_fun,
        x0=theta0,
        method='SLSQP',
        options={'ftol': 1e-10, 'maxiter': 500}, # Using the higher maxiter
        jac=jac_fun,
        bounds=bounds
    )
    
    if result.success:
        optimized_angles = result.x
        
        # 1. Compute and Store POVMs (Efficiently)
        # We calculate Unnormalized once for printing, and Normalized for storage/printing
        pi_unnorm = Pi_N_Unnorm(optimized_angles, fixed_phi)
        pi_norm = Pi_N_Norm(optimized_angles, fixed_phi)
        
        # 2. Extract or Calculate Metrics based on objective
        target_state = Rphi(fixed_phi) @ H
        if objective == 'Fidelity':
            final_fidelity = -result.fun
            final_td = Td(pi_norm, target_state)
        else: # TraceNorm
            final_td = result.fun
            final_fidelity = Fid(pi_norm, target_state)

        # 3. Attach Metadata to result object
        result.fidelity = final_fidelity
        result.trace_distance = final_td
        result.pi_norm = pi_norm
        
        # 4. Detailed Summary Printout
        print("---------------------------------------------------------------")
        print(f"RESULTS FOR N = {N} | Objective: {objective}")
        print(f"Optimization Status: {result.message}")
        print(f"Number of Iterations: {result.nit}")
        print(f"Function Evaluations: {result.nfev}")
        print(f"Optimal Angles (theta): \n{optimized_angles}")
        print(f"Final Fidelity: {result.fidelity:.8f}")
        print(f"Final Trace Distance: {result.trace_distance:.8f}")
        
        print("\n[ Unnormalized POVM (Pi_N) ]")
        print(pi_unnorm)
        
        print("\n[ Normalized POVM (Pi_N_Norm) ]")
        print(pi_norm)
        print("---------------------------------------------------------------")
        
    else:
        print(f"!!! Optimization FAILED for N={N}: {result.message} !!!")
        
    return result

#~V5: Optimization OF BETA~#
def run_optimization_B(N, fixed_phi=0.0):
    """
    Sets up and runs the SLSQP optimization for N scattering events (thetas).
    
    Args:
        N (int): The number of scattering events (angles to optimize).
        fixed_phi (float): The fixed value of the rotation angle phi. Not
        involved in optimization. Just to get rotated POVMs.
    """
    
    print(f"\n--- Running Optimization of beta for N = {N} Events) ---")

    # --- Setup Objective Function as -beta and Jacobian as gradiant ---
    
    def obj_fun(theta_vec):
        return -np.real(beta_N(theta_vec))  #beta is real
    jac_fun=grad(obj_fun)

    # --- Setup N-dimensional Inputs ---
    
    # Inputs are only the N theta angles
    bounds = [(0, np.pi)] * N
    np.random.seed(42) 
    theta0 = np.array([np.pi/4 + np.random.uniform(-0.5, 0.5) for _ in range(N)])

    print(f"Initial Guess (theta0): {theta0}")

    # --- Run the Minimization ---
    
    result = minimize(
        fun=obj_fun,
        x0=theta0,
        method='SLSQP',
        options={'ftol': 1e-10, 'maxiter': 1000}, # 5000 for N=500, 1000 for N=100. Increased from 500 to handle high N
        jac=jac_fun,
        bounds=bounds
    )
    
    if result.success:
        optimized_angles = result.x
        
        # 1. Compute and Store POVMs (Efficiently)
        # We calculate Unnormalized once for printing, and Normalized for storage/printing
        pi_unnorm = Pi_N_Unnorm(optimized_angles, fixed_phi)
        pi_norm = Pi_N_Norm(optimized_angles, fixed_phi)
        
        # 2. Calculate Fid and Td from beta   
        final_beta=beta_N(optimized_angles)        
        final_fidelity = FidB(final_beta)
        final_td = TdB(final_beta)

        # 3. Attach Metadata to result object
        result.beta=final_beta
        result.fidelity = final_fidelity
        result.trace_distance = final_td
        result.pi_norm = pi_norm
        
        # 4. Detailed Summary Printout
        print("---------------------------------------------------------------")
        print(f"RESULTS FOR N = {N} | Objective: beta")
        print(f"Optimization Status: {result.message}")
        print(f"Number of Iterations: {result.nit}")
        print(f"Function Evaluations: {result.nfev}")
        print(f"Optimal Angles (theta): \n{optimized_angles}")
        print(f"beta: {result.beta:.8f}")
        print(f"Final Fidelity: {result.fidelity:.8f}")
        print(f"Final Trace Distance: {result.trace_distance:.8f}")
        
        print("\n[ Unnormalized POVM (Pi_N) ]")
        print(pi_unnorm)
        
        print("\n[ Normalized POVM (Pi_N_Norm) ]")
        print(pi_norm)
        print("---------------------------------------------------------------")
        
    else:
        print(f"!!! Optimization FAILED for N={N}: {result.message} !!!")
        
    return result


#~Function For How Energy Reduces~#
def energy_decay(theta_vec, E0=1.0):
    """
    Calculates the energy after each scattering event.
    
    Args:
        theta_vec (array): The optimal angles [theta_1, theta_2, ... theta_N].
        E0 (float): Initial energy. Default is 1.0.
        
    Returns:
        list: Energy values starting from E0 followed by energy after each event.
    """
    energies = [E0]
    current_E = E0
    
    for theta in theta_vec:
        current_E = Eout(current_E,theta)
        energies.append(current_E)
        
    return energies

r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ░█▀▄░█▀▀░█░░░█░░░░░▀█▀░█▀▀░█▀▀░▀█▀░░░█▀▀░█▀█░█▀▄░░░█▀█░█▀█░█▀▄░█▀█░░░░░█▀█░█▀▀
# ░█▀▄░█▀▀░█░░░█░░░░░░█░░█▀▀░▀▀█░░█░░░░█▀▀░█░█░█▀▄░░░█▀▀░█▀█░█▀▄░█▀█░▄▄▄░█▀▀░▀▀█
# ░▀▀░░▀▀▀░▀▀▀░▀▀▀░░░░▀░░▀▀▀░▀▀▀░░▀░░░░▀░░░▀▀▀░▀░▀░░░▀░░░▀░▀░▀░▀░▀░▀░░░░░▀░░░▀▀▀

    - Probability for phi1, phi2 outcomes
    - Expectation values in CHSH inequality
    - Computes Maximal Violation of Bell Inequalities at phi=Pi/8
    - Plots S(phi) from the maximal violation. Can show that Tphi is related to
      Tun_N by a rotation matrix M(phi), so no need to compute whole function.
    - V5: SPLOT USING THE BETA OPTIMIZATION
    - V7: New function to find Dtheta, for a given N, where S-->2.
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

#~Probability at phi1 and phi2~#
# No need to normalize by sigma_tot here, will cancel in ratio of exp vals
def PV(theta_vec,phi1,phi2):
    Pi2mode=np.kron(Pi_N_Unnorm(theta_vec,phi1),Pi_N_Unnorm(theta_vec,phi2))
    
    return Pr(rhoBell,Pi2mode)

#~Expectation value~#
def EV(result_object, N, phi1, phi2):
    """
    Extracts the optimal angles from the minimization result and computes expectation values

    Args:
        result_object (OptimizeResult): The result object returned run_optimization_N.
        N: The number of scattering events.

    Returns:
        float: The expectation value E(a,b), or None if optimization failed.
    """

    if not result_object.success:
        print(f"\n[ Computation Failed ] Optimization for N={N} was unsuccessful.")
        return None

    # 1. Extract the optimal angle vector (x)
    optimal_angles = result_object.x
    
    # Optional safety check (usually redundant if N is passed correctly)
    if len(optimal_angles) != N:
         print(f"[ Error ] Expected {N} angles, but found {len(optimal_angles)} in result.")
         return None

    # 2. Compute Expectation value
    Eab_num=PV(optimal_angles,phi1,phi2)+PV(optimal_angles,phi1+np.pi/2,phi2+np.pi/2)+\
        -PV(optimal_angles,phi1,phi2+np.pi/2)-PV(optimal_angles,phi1+np.pi/2,phi2)
    Eab_denom=PV(optimal_angles,phi1,phi2)+PV(optimal_angles,phi1+np.pi/2,phi2+np.pi/2)+\
        PV(optimal_angles,phi1,phi2+np.pi/2)+PV(optimal_angles,phi1+np.pi/2,phi2)
    
    Eab=Eab_num/Eab_denom

    print(f"\n--- Expectation Value (N={N}, phi1={phi1}, and phi2={phi2}) ---")
    print(f"Optimal Angle Vector (x): {optimal_angles}")
    print(f"Eab: {Eab:.6f}")
    print("-" * 40)

    return Eab

#~CHSH Function~#
def S_CHSH(result_object, N, phi):
    """
    Computes the full CHSH-Bell function (S_CHSH) based on the optimal angles 
    from the minimization result and a single parameter phi.

    The calculation is: S = |E(0, -phi) - E(0, -3*phi) + E(2*phi, -phi) + E(2*phi, -3*phi)|
    
    The minus signs are chosen to match the function |3Cos(2phi)-Cos(6phi)|

    Args:
        result_object (OptimizeResult): The result object returned run_optimization_N.
        N (int): The number of scattering events.
        phi (float): The single angle parameter defining the measurement settings.

    Returns:
        float: The S_CHSH value, or None if the underlying optimization failed.
    """

    if not result_object.success:
        print(f"\n[ Computation Failed ] Optimization for N={N} was unsuccessful. Cannot compute S_CHSH.")
        return None

    # --- 1. Compute the Four Expectation Values ---
    
    # E1: E(A, B) where A=0, B=-phi
    E1 = EV(result_object, N, phi1=0.0, phi2=-phi)

    # E2: E(A, B') where A=0, B'=-3*phi
    E2 = EV(result_object, N, phi1=0.0, phi2=-3 * phi)
    
    # E3: E(A', B) where A'=2*phi, B=-phi
    E3 = EV(result_object, N, phi1=2 * phi, phi2=-phi)

    # E4: E(A', B') where A'=2*phi, B'=-3*phi
    E4 = EV(result_object, N, phi1=2 * phi, phi2=-3 * phi)
    
    # Check for EV function failure (e.g., if a denominator was zero)
    if any(E is None for E in [E1, E2, E3, E4]):
        print("S_CHSH calculation aborted due to failed EV computation.")
        return None

    # --- 2. Combine to find S_CHSH ---
    S_chsh_value = np.abs(E1 - E2 + E3 + E4)

    print(f"\n=== S_CHSH Result (N={N}, phi={phi:.4f}) ===")
    print(f"E(0, -phi)    (E1) = {E1:.6f}")
    print(f"E(0, -3*phi)  (E2) = {E2:.6f}")
    print(f"E(2*phi, -phi) (E3) = {E3:.6f}")
    print(f"E(2*phi, -3*phi) (E4) = {E4:.6f}")
    print(f"S_CHSH Value: {S_chsh_value:.6f}")
    print("=" * 40)

    return np.abs(S_chsh_value)

#~S(phi) plotting function and save data~#
def S_plot(Nmax, phi_vec, objective='Fidelity', fixed_phi_opt=np.pi/8, save_filename=None):
    """
    Computes S(phi) for various N by scaling by the maximum S_CHSH value found 
    via optimization, and plots the absolute value.
    """
    
    # --- 0. Setup Data Storage Lists ---
    optimal_angles_list = []
    S_max_values = []
    Fidelity_values = []
    TraceNorm_values = []
    Energy_histories = [] 
    
    # 1. Setup the figure
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(12, 6)) # Widened for colorbar
    colors = mpl.colormaps['turbo']
    Nmid = 5 # Midpoint for colormap
    
    # TwoSlopeNorm for the nonlinear scaling maps 1 -> 0.0, Nmid -> 0.5, 
    #a nd Nmax -> 1.0 automatically.
    if Nmax > Nmid:
        norm = mpl.colors.TwoSlopeNorm(vmin=1, vcenter=Nmid, vmax=Nmax)
    else:
        norm = mpl.colors.Normalize(vmin=1, vmax=max(2, Nmax))
    
    # 2. S(phi) if we had perfect polarization measurements
    def S_perfect(phi):
        return (3 * np.cos(2 * phi) - np.cos(6 * phi)) 

    # 3. Iterate through N = 1 to Nmax
    for N in range(1, Nmax + 1):
        color_val = colors(norm(N))
            
        # Define max and min linewidth for the scaling
        max_width = 3.0
        min_width = 1.0
        
        if Nmax <= 1:
            # Case: Only one curve is plotted
            current_linewidth = max_width
        else:
            # Normalize N from 1 to Nmax to a 0 to 1 range
            # The denominator (Nmax - 1) handles the full range of scaling
            N_scaled = (N - 1) / (Nmax - 1)
            
            # Calculate the line width: it scales linearly from max_width (N=1)
            # down to min_width (N=Nmax).
            current_linewidth = max_width - N_scaled * (max_width - min_width)
        # ------------------------------------------------------------------
        
        print(f"\nProcessing N={N}...")
        
        
        try:
            # Run optimization (the objective can be 'Fidelity' or 'TraceNorm')
            opt_result = run_optimization_N(N=N, objective=objective, fixed_phi=0.0)
            
            if not opt_result.success:
                continue
    
            # --- DATA Saving ---
            optimal_angles_list.append(opt_result.x)
            Fidelity_values.append(opt_result.fidelity)
            TraceNorm_values.append(opt_result.trace_distance)
            
            # Energy decay needs to be calculated
            e_path = energy_decay(opt_result.x, E0=1.0)
            Energy_histories.append(e_path) # Stores the list of N+1 energy values
    
            # S_CHSH still needs to be called 
            S_max = S_CHSH(opt_result, N, phi=fixed_phi_opt)
            
            if S_max is None: continue
            
            S_max = np.abs(S_max)
            S_max_values.append(S_max)
            
            # --- B. Calculate the scaled curve ---
            scaled_amplitude = S_max / (2 * np.sqrt(2)) 
            S_curve = np.abs(scaled_amplitude * S_perfect(phi_vec))
            
            # --- C. Plot the curve ---
            plt.plot(phi_vec, S_curve,
                     color=color_val, 
                     linewidth=current_linewidth, 
                     alpha=0.85, 
                     zorder=2)


        except Exception as e:
            print(f"An error occurred while processing N={N}: {e}")
            continue

    # --- 4. Plot the Bounds ---
    
    # Tsirelson's Bound
    plt.axhline(y=2 * np.sqrt(2), color='black', linestyle='--', label=r"Tsirelson's Bound", linewidth=1.5, zorder=3)

    # LHV Bound
    plt.axhline(y=2.0, color='black', linestyle='-', label=r"LHV Bound", linewidth=1.5, zorder=3)
    
    # --- 5. Final Plot Aesthetics ---
    
    # --- COLORBAR ---
    # Create a 'ScalarMappable' to link the colormap and the norm
    sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([]) # Dummy array for the mappable
    
    # Create the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02, aspect=30)
    cbar.set_label(r'Number of Scattering Events ($N$)', fontsize=12, rotation=270, labelpad=15)
    
    # Colorbar Ticks
    # Spacing for the ticks
    tick_step = 5 if Nmax > 50 else 1
    # Tick locations
    # We use a 'set' to ensure 1, 2, Nmid, and Nmax are always present, 
    # then add the multiples of tick_step (5, 10, 15...)
    milestones = {1, 2, Nmid, Nmax}
    grid_ticks = set(range(tick_step, Nmax + 1, tick_step))
    all_ticks = sorted(list(milestones | grid_ticks))
    cbar.set_ticks(all_ticks)
    # Create the labels
    # We only want text for our specific milestones
    tick_labels = []
    for t in all_ticks:
        if t in [1, 2, Nmid, Nmax]:
            tick_labels.append(str(t))
        else:
            # This keeps the little line (tick) but removes the number text
            tick_labels.append("") 
    
    cbar.set_ticklabels(tick_labels)
    
    # --- REST OF THE PLOT
    plt.xlim(np.min(phi_vec), np.max(phi_vec))
    plt.ylim(0, 2 * np.sqrt(2) * 1.05) 

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', alpha=1.0, zorder=1)
    plt.grid(True, which='minor', linestyle=':', alpha=1.0, zorder=1)
    
    # Title and Labels 
    plt.title(r'CHSH Inequality Function $|S(\phi)|$ for $N=1$ to $N=' + str(Nmax) + r'$', fontsize=16)
    plt.xlabel(r'Azimuthal Angle $\phi$ (radians)', fontsize=14) 
    plt.ylabel(r'$|S(\phi)|$', fontsize=14) 

    # Dynamic Ticks (using the dynamic labels)
    max_phi = np.max(phi_vec)
    tick_map = {
        0: r'$0$', np.pi/4: r'${\pi}/{4}$', np.pi/2: r'${\pi}/{2}$', 3*np.pi/4: r'${3\pi}/{4}$', 
        np.pi: r'$\pi$', 5*np.pi/4: r'${5\pi}/{4}$', 3*np.pi/2: r'${3\pi}/{2}$', 
        7*np.pi/4: r'${7\pi}/{4}$', 2*np.pi: r'$2\pi$',
    }
    
    filtered_ticks = {loc: label for loc, label in tick_map.items() if loc <= max_phi + 1e-9}
    
    # Tick labels
    plt.xticks(list(filtered_ticks.keys()), list(filtered_ticks.values())) 
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=12, direction='in')
    ax.tick_params(axis='y', labelsize=12, direction='in')
    
    # Legend
    plt.legend(loc='lower center', frameon=True, fancybox=True, shadow=True, fontsize=14) 
    
    plt.tight_layout()

    # --- SAVE THE PLOT AS SVG ---
    if save_filename:
        # bbox_inches='tight' so no labels are cut off.
        plt.savefig(save_filename, format='svg', bbox_inches='tight')
        print(f"Plot saved successfully as {save_filename}")

    plt.show()
    
    # --- RETURN DATA ---
    return (optimal_angles_list, S_max_values, Fidelity_values, 
            TraceNorm_values, Energy_histories)

#~V5: S(phi) plotting function and save data USING BETA OPTIMIZATION~#
def S_plotB(Nmax, phi_vec, fixed_phi_opt=np.pi/8, save_filename=None):
    """
    Computes S(phi) for various N by scaling by the maximum S_CHSH value found 
    via BETA optimization, and plots the absolute value.
    """
    
    # --- 0. Setup Data Storage Lists ---
    optimal_angles_list = []
    beta_values = []
    S_max_values = []
    Fidelity_values = []
    TraceNorm_values = []
    Energy_histories = [] 
    
    # 1. Setup the figure
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(12, 6)) # Widened for colorbar
    colors = mpl.colormaps['turbo']
    Nmid = 5 # Midpoint for colormap
    
    # TwoSlopeNorm for the nonlinear scaling maps 1 -> 0.0, Nmid -> 0.5, 
    #a nd Nmax -> 1.0 automatically.
    if Nmax > Nmid:
        norm = mpl.colors.TwoSlopeNorm(vmin=1, vcenter=Nmid, vmax=Nmax)
    else:
        norm = mpl.colors.Normalize(vmin=1, vmax=max(2, Nmax))
    
    # 2. S(phi) if we had perfect polarization measurements
    def S_perfect(phi):
        return (3 * np.cos(2 * phi) - np.cos(6 * phi)) 

    # 2. Iterate through N = 1 to Nmax
    for N in range(1, Nmax + 1):
        color_val = colors(norm(N))
            
        # Define max and min linewidth for the scaling
        max_width = 3.0
        min_width = 1.0
        
        if Nmax <= 1:
            # Case: Only one curve is plotted
            current_linewidth = max_width
        else:
            # Normalize N from 1 to Nmax to a 0 to 1 range
            # The denominator (Nmax - 1) handles the full range of scaling
            N_scaled = (N - 1) / (Nmax - 1)
            
            # Calculate the line width: it scales linearly from max_width (N=1)
            # down to min_width (N=Nmax).
            current_linewidth = max_width - N_scaled * (max_width - min_width)
        # ------------------------------------------------------------------
        
        print(f"\nProcessing N={N}...")
        
        
        try:
            # Run optimization (the objective IS BETA NOW)
            opt_result = run_optimization_B(N=N, fixed_phi=0.0)
            
            if not opt_result.success:
                continue
    
            # --- DATA Saving ---
            optimal_angles_list.append(opt_result.x)
            beta_values.append(opt_result.beta)
            Fidelity_values.append(opt_result.fidelity)
            TraceNorm_values.append(opt_result.trace_distance)
            
            # Energy decay needs to be calculated
            e_path = energy_decay(opt_result.x, E0=1.0)
            Energy_histories.append(e_path) # Stores the list of N+1 energy values
    
            # V5: CALL SB NOW
            S_max = SB(opt_result.beta, phi=fixed_phi_opt)
            
            if S_max is None: continue
            
            S_max_values.append(S_max)
            
            # --- B. Calculate the scaled curve ---
            scaled_amplitude = S_max / (2 * np.sqrt(2)) 
            S_curve = np.abs(scaled_amplitude * S_perfect(phi_vec))
            
            # --- C. Plot the curve ---
            plt.plot(phi_vec, S_curve,
                      color=color_val, 
                      linewidth=current_linewidth, 
                      alpha=0.85, 
                      zorder=2)


        except Exception as e:
            print(f"An error occurred while processing N={N}: {e}")
            continue

    # --- 4. Plot the Bounds ---
    
    # Tsirelson's Bound
    plt.axhline(y=2 * np.sqrt(2), color='black', linestyle='--', label=r"Tsirelson's Bound", linewidth=1.5, zorder=3)

    # LHV Bound
    plt.axhline(y=2.0, color='black', linestyle='-', label=r"LHV Bound", linewidth=1.5, zorder=3)
    
    # --- 5. Final Plot Aesthetics ---
    
    # --- COLORBAR ---
    # Create a 'ScalarMappable' to link the colormap and the norm
    sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([]) # Dummy array for the mappable
    
    # Create the colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02, aspect=30)
    cbar.set_label(r'Number of Scattering Events ($N$)', fontsize=12, rotation=270, labelpad=15)
    
    
    # Colorbar Ticks
    # Spacing for the ticks
    tick_step = 5 if Nmax > 50 else 1
    # Tick locations
    # We use a 'set' to ensure 1, 2, Nmid, and Nmax are always present, 
    # then add the multiples of tick_step (5, 10, 15...)
    milestones = {1, 2, Nmid, Nmax}
    grid_ticks = set(range(tick_step, Nmax + 1, tick_step))
    all_ticks = sorted(list(milestones | grid_ticks))
    cbar.set_ticks(all_ticks)
    # Create the labels
    # We only want text for our specific milestones
    tick_labels = []
    for t in all_ticks:
        if t in [1, 2, Nmid, Nmax]:
            tick_labels.append(str(t))
        else:
            # This keeps the little line (tick) but removes the number text
            tick_labels.append("") 
    
    cbar.set_ticklabels(tick_labels)
    
    
    # --- REST OF THE PLOT
    plt.xlim(np.min(phi_vec), np.max(phi_vec))
    plt.ylim(0, 2 * np.sqrt(2) * 1.05) 

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', alpha=1.0, zorder=1)
    plt.grid(True, which='minor', linestyle=':', alpha=1.0, zorder=1)
    
    # Title and Labels 
    plt.title(r'CHSH Inequality Function $|S(\phi)|$ for $N=1$ to $N=' + str(Nmax) + r'$', fontsize=16)
    plt.xlabel(r'Azimuthal Angle $\phi$ (radians)', fontsize=14) 
    plt.ylabel(r'$|S(\phi)|$', fontsize=14) 

    # Dynamic Ticks (using the dynamic labels)
    max_phi = np.max(phi_vec)
    tick_map = {
        0: r'$0$', np.pi/4: r'${\pi}/{4}$', np.pi/2: r'${\pi}/{2}$', 3*np.pi/4: r'${3\pi}/{4}$', 
        np.pi: r'$\pi$', 5*np.pi/4: r'${5\pi}/{4}$', 3*np.pi/2: r'${3\pi}/{2}$', 
        7*np.pi/4: r'${7\pi}/{4}$', 2*np.pi: r'$2\pi$',
    }
    
    filtered_ticks = {loc: label for loc, label in tick_map.items() if loc <= max_phi + 1e-9}
    
    # Tick labels
    plt.xticks(list(filtered_ticks.keys()), list(filtered_ticks.values())) 
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=12, direction='in')
    ax.tick_params(axis='y', labelsize=12, direction='in')
    
    # Legend
    plt.legend(loc='lower center', frameon=True, fancybox=True, shadow=True, fontsize=14) 
    
    plt.tight_layout()

    # --- SAVE THE PLOT AS SVG ---
    if save_filename:
        # bbox_inches='tight' so no labels are cut off.
        plt.savefig(save_filename, format='svg', bbox_inches='tight')
        print(f"Plot saved successfully as {save_filename}")

    plt.show()
    
    # --- RETURN DATA, INCLUDING BETA VALUES ---
    return (optimal_angles_list, beta_values, S_max_values, Fidelity_values, 
            TraceNorm_values, Energy_histories)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# ░█▀▀░█░█░█▀▀░█▀▀░█░█░▀█▀░▀█▀░█▀█░█▀█
# ░█▀▀░▄▀▄░█▀▀░█░░░█░█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀
# Optimize via Fidelity/Trace Norm OR optimize the beta function (faster)
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# ------------------------------------------
#~Optimization of theta angles~#
# ------------------------------------------
#1. Different Scattering Events, Different Objective, Fid, Td, beta
#N=3 scattering events:
# run_optimization_N(N=3, objective='Fidelity')
# run_optimization_N(N=3, objective='TraceNorm')
# run_optimization_B(N=2)
#N=5 scattering events:
# run_optimization_N(N=5, objective='Fidelity')
# run_optimization_N(N=5, objective='TraceNorm')
# run_optimization_B(N=5)
#2. Different phi values
# N_value = 5
# opt_result = run_optimization_N(N=N_value, objective='Fidelity', fixed_phi=0.0)
# opt_result = run_optimization_N(N=N_value, objective='TraceNorm', fixed_phi=0.0)
# opt_result = run_optimization_B(N=N_value, fixed_phi=0.0)

# ------------------------------------------
#~How energy decays over N scattering events~#
# ------------------------------------------
# N_value=3
# 1. Fidelity or trace norm
# opt_result = run_optimization_N(N=N_value, objective='Fidelity', fixed_phi=0.0)
# if opt_result.success:
#     energy_history = energy_decay(opt_result.x, E0=1.0)
#     final_energy = energy_history[-1]
#     print(f"Initial Energy: {energy_history[0]}")
#     print(f"Energy after {N_value} events: {final_energy:.6f}")
# 2. beta function
# opt_result = run_optimization_B(N=N_value, fixed_phi=0.0)
# if opt_result.success:
#     energy_history = energy_decay(opt_result.x, E0=1.0)
#     final_energy = energy_history[-1]
#     print(f"Initial Energy: {energy_history[0]}")
#     print(f"Energy after {N_value} events: {final_energy:.6f}")

# ------------------------------------------
#~Max value of the CHSH function at pi/8~#
# ------------------------------------------
# 1. Fid or trace norm
# N_value = 10
# opt_result = run_optimization_N(N=N_value, objective='Fidelity')
# Eval = EV(opt_result, N_value, 0, 0)
# S_opt=S_CHSH(opt_result, N_value, np.pi/8)
# 2. beta 
N_value = 10
opt_result = run_optimization_B(N=N_value)
S_opt = SB(opt_result.beta, np.pi/8)
print(f"S_opt: {S_opt:.8f}")

# ------------------------------------------
#~Plotting CHSH vs phi~#
# ------------------------------------------
# 1. Fid or Trace norm
# phi_angles = np.linspace(0, 1 * np.pi, 100)
# S_plot(Nmax=7, phi_vec=phi_angles)
# S_plot(Nmax=7, phi_vec=phi_angles, objective='TraceNorm')
# 2. beta
# phi_angles = np.linspace(0, 1 * np.pi, 100)
# S_plotB(Nmax=10, phi_vec=phi_angles)

# --------------------------------------------------
#~Plotting CHSH vs phi and saving figs and or data~#
# --------------------------------------------------
# 1. Fid or Trace norm
# phi_angles = np.linspace(0, 1 * np.pi, 100)
# (Angles, Smax, Fid_data, 
#   TD_data, E_paths) = S_plot(Nmax=20, phi_vec=phi_angles, 
#                             save_filename="CHSH_and_Energy_N20.svg")
# angles_to_save = np.array(Angles, dtype=object)
# energies_to_save = np.array(E_paths, dtype=object)
# np.savez_compressed(
#     'chsh_complete_data_N20.npz',
#     optimal_angles=angles_to_save,
#     energy_paths=energies_to_save,
#     S_CHSH_max=Smax,
#     optimized_fidelity=Fid_data,
#     trace_distances=TD_data,
#     allow_pickle=True
# )
# 2. beta optimization (can check Smax/beta values =2sqrt(2))
# takes about 4 mins 18s to run 50
# increase phi_angles points to get smoother plot
# phi_angles = np.linspace(0, 1 * np.pi, 500)
# (Angles, beta_data, Smax, Fid_data, 
#   TD_data, E_paths) = S_plotB(Nmax=10, phi_vec=phi_angles, 
#                             save_filename="CHSH_and_Energy_N10.svg")
# angles_to_save = np.array(Angles, dtype=object)
# energies_to_save = np.array(E_paths, dtype=object)
# np.savez_compressed(
#     'chsh_complete_data_N10.npz',
#     optimal_angles=angles_to_save,
#     energy_paths=energies_to_save,
#     betas=beta_data,
#     S_CHSH_max=Smax,
#     optimized_fidelity=Fid_data,
#     trace_distances=TD_data,
#     allow_pickle=True
# )



