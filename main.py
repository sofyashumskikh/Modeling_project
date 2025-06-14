import pybullet as p
import time
import pybullet_data
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from control.matlab import place, lqr, ctrb
from numpy.linalg import matrix_rank as rank

guiFlag = False
dt = 1/240
th0 = np.pi - 0.1
thd = np.pi
g = 10
L = 0.8
m = 1

A = np.array([[0, 1], [g/L, 0]])
B = np.array([[0], [1/(m*L*L)]])

U = ctrb(A, B)
if rank(U) != A.shape[0]:
    print("Ошибка: система неуправляема!")
    exit()
else:
    print("Система управляема, можно применить модальный синтез")

# Модальный синтез
poles = np.array([-2, -3])
K = -place(A, B, poles)
print("Матрица обратной связи K =", K)

physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("simple.urdf.xml", useFixedBase=True)

p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5
logTime = np.arange(0, maxTime, dt)
sz = len(logTime)
logThetaSim = np.zeros(sz)
logVelSim = np.zeros(sz)
logTauSim = np.zeros(sz)
idx = 0
max_tau = 20

for t in logTime:
    th = p.getJointState(boxId, 1)[0]
    vel = p.getJointState(boxId, 1)[1]
    logThetaSim[idx] = th

    tau_fb_lin = m * L**2 * (g/L * np.sin(th))

    e = th - thd
    tau_modal = K[0, 0] * e + K[0, 1] * vel

    tau = tau_fb_lin + tau_modal
    tau = np.clip(tau, -max_tau, max_tau)
    logTauSim[idx] = tau

    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=tau, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()

    vel = p.getJointState(boxId, 1)[1]
    logVelSim[idx] = vel
    idx += 1

    if guiFlag:
        time.sleep(dt)

p.disconnect()

# Визуализация
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(logTime, logThetaSim, 'b', label="Угол (симуляция)")
plt.plot([logTime[0], logTime[-1]], [thd, thd], 'r--', label="Целевое положение")
plt.ylabel("Угол (рад)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(logTime, logVelSim, 'g', label="Скорость")
plt.ylabel("Скорость (рад/с)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(logTime, logTauSim, 'r', label="Управляющий момент")
plt.ylabel("Момент (Н·м)")
plt.xlabel("Время (с)")
plt.grid(True)
plt.legend()

plt.suptitle("Стабилизация маятника в верхнем положении (θ = π)")
plt.show()