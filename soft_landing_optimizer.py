# Implemented soft landing powered decent guidance optimization problem
# based on the paper   
# Lossless Convexification of Non-Convex Control Bound and Pointing
# Constraints of the Soft Landing Optimal Control Problem.
# http://www.larsblackmore.com/iee_tcst13.pdf     
# B. Acikmese, J. M. Carson III, and L. Blackmore.
# IEEE Transactions on Control Systems Technology, Volume 21, Issue 6 (2013)
#
# Author: ravi_ram

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# initial state vector
# r0 = 2400, 450, −330 m, ṙ0 = −10, −40, 10 m/s
x0 = np.array([2400, 450, -330, -10, -40, 10]).T #x0 = [r, ṙ]).T

# landing target coordinates (2D, alt=0 => x=0)
q  = np.array([0, 0])      

# simulations which make use of an example with the following properties:
# m0 = 2000 kg, mf = 300 kg,
# α > 0 is a constant that describes the fuel consumption (mass depletion) rate.
# α = 5 × 10^−4 s/m 
m0 = 2000               # initial mass (kg)
mf = 300                # fuel mass (kg)
alpha = 5e-4            # α - fuel consumption rate (s/m)

# The thrust limits coincide with a minimum and maximum throttle level of 20% and 80%,
# respectively.
# ρ1 = 0.2 × Tmax , ρ2 = 0.8 × Tmax , Tmax = 24000 N,
# where Tmax is the maximum thrust magnitude.
Tmax = 24000            # max thrust (N)
rho1 = 0.2 * Tmax       # ρ1, lower bound thrust (N)
rho2 = 0.8 * Tmax       # ρ2, upper bound thrust (N)

# simulation time
tf = 50  # end time (s)
dt = 1   # time interval (s)
N = int(tf / dt)

# log of initial mass and final mass
zi = np.log(m0)
zf = np.log(m0 - mf)

# glide slope angle γgs = 30° (minimum glide-slope angle from the ground plane)
glidelslope_angle  = 30
gamma_tan = np.tan(np.deg2rad(glidelslope_angle))
# thrust angle θ = 120°
theta_deg = 120
theta_cos = np.cos(np.deg2rad(theta_deg) )
# maximum velocity of 90 m/s
velocity_max = 90

# planet (mars) parameters
# ω = (ω1 , ω2 , ω3 ) ∈ R3 is the vector of planet’s constant angular velocity,
# ω = (2.53 × 10^−5, 0, 6.62 × 10^−5) rad/s
# g = (−3.71, 0, 0 ) m/s^2, g ∈ R3 is the constant gravity vector
ω = np.array([2.53e-5, 0, 6.62e-5]).T     # rotation angular velocity
g = np.array([-3.71, 0, 0]).T             # gravitational acceleration

# unit vectors
e1 = np.array([1,0,0]).T
e2 = np.array([0,1,0]).T
e3 = np.array([0,0,1]).T
# E matrix 
E = np.array([e2.T, e3.T])

# S matrix
S = np.array([ [0, -ω[2], ω[1]],  [ω[2], 0, -ω[0]], [ω[1], ω[0], 0]  ])
# A matrix
A = np.block([ [np.zeros((3, 3)), np.eye(3)],  [-S**2, -2*S] ])
# B matrix
B = np.block([ [np.zeros((3, 3))], [np.eye(3)] ])


# Problem 3 (Convex Relaxed Minimum Landing Error Problem)
def solve_minimum_err(dt, N, x0):
    # cvxpy variables
    x  = cp.Variable((6, N))     # [pos(3), vel(3)]
    z  = cp.Variable((1, N))     # ln(m)
    u  = cp.Variable((3, N))     # thrust
    gamma  = cp.Variable((1, N)) # slack variable (Γ)

    # minimize the error between the goal and the landing position
    # subject to (5), (7), (8), (9), and (17), (18), (19)
    objective = cp.Minimize( cp.norm(E@x[:3, N-1] - q) )
    # constraints
    constraints = []   
    constraints = set_constraints(constraints, x, z, u, gamma)
    
    # create the problem
    problem = cp.Problem(objective, constraints)
    # solve
    problem.solve()
    return problem.status, x, u

# Problem 4 (Convex Relaxed Minimum Fuel Problem)
# dP3 is the final optimal position from Problem 3
def solve_fuel_opt(dt, N, x0, dP3):
    # cvxpy variables
    x  = cp.Variable((6, N))     # [pos(3), vel(3)]
    z  = cp.Variable((1, N))     # ln(m)
    u  = cp.Variable((3, N))     # thrust
    gamma  = cp.Variable((1, N)) # slack variable (Γ)
    
    # minimize Integral [ Γ(t) * dt ] 0 to tf
    # subject to (5), (7), (8), (9), (17), (18), (19) and (20)
    objective = cp.Minimize( cp.sum(gamma) * dt )
    # constraints
    constraints = []   
    constraints = set_constraints(constraints, x, z, u, gamma)
    
    # || E * r(tf) − q || ≤  dP3                                ------ (20)
    constraints.append( cp.norm(E @ x[:3, N-1] - q) <= dP3 )
    
    # create the problem
    problem = cp.Problem(objective, constraints)
    # solve
    problem.solve()   
    return problem.status, x, u

# common constraints for both problems
def set_constraints(constraints, x, z, u, gamma):
    # boundary constraints                                       ------ (8)
    constraints.append(x[:,0] == x0)              # initial velocity and position
    # m(0) = m0 , m(tf) ≥ m0 − mf > 0                            ------ (7)
    constraints.append(z[0, 0] == zi)             # initial mass
    constraints.append(z[0, N-1] >= zf)           # final empty mass   
    # e1.T r(tf) = 0, ṙ(tf) = 0.                                 ------ (9)
    constraints.append(e1.T @ x[:3, N-1] == 0)    # final altitude = 0 
    constraints.append(e1.T @ x[3:, N-1] == 0)    # final velocity = 0
    
    constraints.append(cp.norm(x[3:6, N-1]) <= velocity_max) # maximum velocity < velocity_max
    
    # dynamic constraints
    # euler integration ( w[t+1] = w[t] + dt * f(t,w[t]) )
    for t in range(N-1):
        # spacecraft dynamics
        # ẋ(t) = (A*x[t] + B*(g+u[t])) * dt       [0, tf]        ------ (17)
        # x[t+1] = x(t) + ẋ(t)
        constraints.append(x[:,t+1] == x[:,t] + (A @ x[:,t] + B @ (g+u[:,t])) * dt )
        # mass depletion dynamics
        # z[t+1] = ż[t] −ασ(t) where ż = −ασ(t)   [0, tf]        ------ (33)
        constraints.append(z[:,t+1] == z[:,t] - alpha * gamma[:,t])

    # thrust constraints   
    # convex upper bound on thrust    || u(t) || ≤ Γ(t)          ------ (34)
    constraints.append(cp.norm(u, axis=0) <= gamma[0,:])
    # convex thrust pointing constraint n̂ Tc(t) ≥ Γ(t) * cos θ   ------ (34)
    # n̂ - unit vector in the vertical direction = e1
    constraints.append(e1 @ u >= gamma[0,:] * theta_cos )
    
    # convex bounds on the slack variable                        ------ (36)
    # ρ1*e^−z0*[ 1 − (z(t) − z0(t)) + (z(t)−z0(t))^2/2 ]
    # ≤ σ(t) ≤
    # ρ2*e^−z0*[ 1 − (z(t) − z0(t)) ]
    # z0(t) = ln(m0 − α*ρ2*t) and m0 is the initial mass of the spacecraft.
    z0 = np.zeros((1, N))
    z0[0] = [np.log(m0 - alpha * rho2 * dt * i) for i in range(N) ]   
    constraints.append( rho1 * np.exp(-zi) * ( 1-(z[0,:] - z0[0,:]) + (z[0,:] - z0[0,:])**2/2 ) <= gamma[0,:] )
    
    # glide slope constraint ensures that the
    # trajectory to the target cannot be too shallow or go subsurface.
    # r_x ≥ tan(γgs) * || r_yz ||
    constraints.append( x[0,:] >= (cp.norm(x[1:3], axis=0)*gamma_tan) )
    
    return constraints

# optimal trajectory
def plot_3d_trajectory(x):  
    Z, Y, X = x.value.T[:,:3].T
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(top=1.1, bottom=-.1)
    ax = fig.add_subplot(121, projection='3d')
    ax.set_aspect('auto')
    ax.view_init(azim=54., elev=22.)
    ax.plot( X, Y, Z )
    ax.set(xlabel='Z axis', ylabel='Y axis', zlabel='X axis')
    ax.set_title('Trajectory')
    plt.show()
    return
    

"""
Algorithm 1 Prioritized powered-descent guidance algorithm
----------------------------------------------------------
1) Solve the relaxed minimum-landing-error guidance problem
   (Problem 4). If no solution exists, return not optimal.
2) Solve the relaxed minimum-fuel guidance problem to specified range
   (Problem 5).
3) Return ft

"""
def solve():
    # Problem 3 (Convex Relaxed Minimum Landing Error Problem):
    status, x, u = solve_minimum_err(dt, N, x0)
    if "optimal" != status:
        print("not optimal")
        return
    
    # Problem 4 (Convex Relaxed Minimum Fuel Problem):
    dP3 = x.value[1:3, N-1]
    status, x, u = solve_fuel_opt(dt, N, x0, dP3)
    
    # plot
    plot_3d_trajectory(x)   
    return

# __main method__ 
if __name__=="__main__":   
    # solve   
    solve()

    
