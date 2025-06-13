import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt

# gui_flag = True
gui_flag = False

dt = 1/240      # simulation time step [seconds]
th0 = 0.1       # starting position [rad]
th_targ = 1.2   # desired position [rad]
T = 3.2         # total time to reach desired position [seconds]

g = 9.81    # [m/s^2]
L = 0.8     # [m]
m = 1       # [kg]

def compute_polynomial_coeffs(T):
    # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    # boundary conditions: s(0) = s'(0) = s''(0) = 0, s(T) = 1, s'(T) = s''(T) = 0

    # from boundary conditions, we have:
    # s(0) = a0 + a1*0 + a2*0^2 + a3*0^3 + a4*0^4 + a5*0^5 = 0  => a0 = 0
    # s'(0) = a1 + 2*a2*0 + 3*a3*0^2 + 4*a4*0^3 + 5*a5*0^4 = 0 => a1 = 0
    # s''(0) = 2*a2 + 6*a3*0 + 12*a4*0^2 + 20*a5*0^3 = 0 => a2 = 0

    # so the polynomial simplifies to (as a0, a1, a2 = 0):
    # s(t) = a3*t^3 + a4*t^4 + a5*t^5

    # therefore, we need to solve the following system of equations:
    # s(T) = a3*T^3 + a4*T^4 + a5*T^5 = 1
    # s'(T) = 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4 = 0
    # s''(T) = 6*a3*T + 12*a4*T^2 + 20*a5*T^3 = 0
    
    # solving the system of equations:
    A_mat = np.array([
        [T**3, T**4, T**5],         # s(T) = 1
        [3*T**2, 4*T**3, 5*T**4],   # s'(T) = 0
        [6*T, 12*T**2, 20*T**3]     # s''(T) = 0
    ])
    b_vec = np.array([1, 0, 0])
    
    coeffs_345 = np.linalg.solve(A_mat, b_vec)
    
    # full polynomial coefficients [a0, a1, a2, a3, a4, a5]
    return np.array([0, 0, 0, coeffs_345[0], coeffs_345[1], coeffs_345[2]])
def evaluate_polynomial(coeffs, t):
    # s(t)
    s = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5
    
    # s'(t)
    s_prime = coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 4*coeffs[4]*t**3 + 5*coeffs[5]*t**4
    
    # s''(t)
    s_prime_prime = 2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3
    
    return s, s_prime, s_prime_prime
def trajectory_interpolation(t, th_start, th_end, T, coeffs):
    if t >= T:
        t = T
    
    s, s_prime, s_prime_prime = evaluate_polynomial(coeffs, t)
    
    # position traj
    theta_des = th_start + (th_end - th_start) * s
    
    # velocity traj
    theta_des_prime = (th_end - th_start) * s_prime
    
    # acceleration traj
    theta_des_prime_prime = (th_end - th_start) * s_prime_prime
    
    return theta_des, theta_des_prime, theta_des_prime_prime

poly_coeffs = compute_polynomial_coeffs(T)
print(f"polynomial coefficients: {poly_coeffs}")

physicsClient = p.connect(p.GUI if gui_flag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
plane_id = p.loadURDF("plane.urdf")
box_id = p.loadURDF("./one_link_pendulum.urdf", useFixedBase=True)

# suppose there's no air drag
p.changeDynamics(box_id, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(box_id, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=box_id, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor (so the pendulum can swing freely)
p.setJointMotorControl2(bodyIndex=box_id, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

max_time = T + 2.0  # run a bit longer than T to see the pendulum's motion
log_time = np.arange(0, max_time, dt)
sz = len(log_time)

# log arrays
log_theta_sim = np.zeros(sz)
log_vel_sim = np.zeros(sz)
log_tau_sim = np.zeros(sz)
log_theta_desired = np.zeros(sz)
log_vel_desired = np.zeros(sz)
log_acc_desired = np.zeros(sz)

# control gains for feedback linearization
kp = 100.0  # position gain
kd = 20.0   # velocity gain

print(f"starting simulation...")
print(f"initial position: {th0:.3f} rad")
print(f"desired position: {th_targ:.3f} rad")
print(f"time to reach target: {T:.1f} seconds")

for idx, t in enumerate(log_time):
    # current state
    th = p.getJointState(box_id, 1)[0]
    vel = p.getJointState(box_id, 1)[1]
    
    # compute desired trajectory
    theta_des, theta_des_prime, theta_des_prime_prime = trajectory_interpolation(t, th0, th_targ, T, poly_coeffs)
    
    # compute errors
    e_pos = th - theta_des
    e_vel = vel - theta_des_prime
    
    # th'' = -g/L*sin(th) + tau/(m*L^2)
    # with feedback linearization: tau = m*L^2*(g/L*sin(th) + u)
    # where u is the new control input: u = theta_des'' - kp*e_pos - kd*e_vel
    
    u = theta_des_prime_prime - kp * e_pos - kd * e_vel
    tau = m * L**2 * (g/L * np.sin(th) + u)
    
    # apply the torque tau
    p.setJointMotorControl2(bodyIndex=box_id, jointIndex=1, force=tau, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()
    
    # log everything
    log_theta_sim[idx] = th
    log_vel_sim[idx] = vel
    log_tau_sim[idx] = tau
    log_theta_desired[idx] = theta_des
    log_vel_desired[idx] = theta_des_prime
    log_acc_desired[idx] = theta_des_prime_prime
    
    if gui_flag:
        time.sleep(dt)

p.disconnect()

# plotting results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# position tracking
ax1.plot(log_time, log_theta_sim, 'b-', label='actual pos', linewidth=2)
ax1.plot(log_time, log_theta_desired, 'r--', label='desired pos', linewidth=2)
ax1.axvline(x=T, color='k', linestyle=':', alpha=0.7, label=f'target Time ({T}s)')
ax1.set_ylabel('position (rad)')
ax1.set_title('position tracking with 5-ord pol traj')
ax1.grid(True, alpha=0.3)
ax1.legend()

# velocity tracking
ax2.plot(log_time, log_vel_sim, 'b-', label='actual vel', linewidth=2)
ax2.plot(log_time, log_vel_desired, 'r--', label='desired vel', linewidth=2)
ax2.axvline(x=T, color='k', linestyle=':', alpha=0.7)
ax2.set_ylabel('velocity [rad/s]')
ax2.set_title('velocity tracking')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# final performance metrics
final_error = abs(log_theta_sim[-1] - th_targ)
print(f"\ntacking overall performance:")
print(f"final position: {log_theta_sim[-1]:.4f} [rad]")
print(f"target position: {th_targ:.4f} [rad]")
print(f"final tracking error: {final_error:.4f} [rad]")
print(f"final velocity: {log_vel_sim[-1]:.4f} [rad/s]")

# polynomial verification
s_0, s_dot_0, s_ddot_0 = evaluate_polynomial(poly_coeffs, 0)
s_T, s_dot_T, s_ddot_T = evaluate_polynomial(poly_coeffs, T)
print(f"\npolynomial verification:")
print(f"at t=0: s={s_0:.6f}, s'={s_dot_0:.6f}, s''={s_ddot_0:.6f}")
print(f"at t=T: s={s_T:.6f}, s'={s_dot_T:.6f}, s''={s_ddot_T:.6f}")
