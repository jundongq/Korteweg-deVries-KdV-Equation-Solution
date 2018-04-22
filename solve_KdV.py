import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#----- Numerical integration of ODE via fixed-step classical Runge-Kutta -----
def RK4Stream(odefunc,TimeSpan,uhat0,nt):
    h = float(TimeSpan[1]-TimeSpan[0])/nt
    w = uhat0
    t = TimeSpan[0]
    while t <= TE:
        w = RK4Step(odefunc, t, w, h)
        t = t+h
        yield t,w

def RK4Step(odefunc, t,w,h):
    k1 = odefunc(t,w)
    k2 = odefunc(t+0.5*h, w+0.5*k1*h)
    k3 = odefunc(t+0.5*h, w+0.5*k2*h)
    k4 = odefunc(t+h,     w+k3*h)
    return w + (k1+2*k2+2*k3+k4)*(h/6.)

#----- Constructing the grid -----
L   = 2.
nx  = 512
x   = np.linspace(0.,L, nx+1)
x   = x[:nx]  

kx1 = np.linspace(0,nx/2-1,nx/2)
kx2 = np.linspace(1,nx/2,  nx/2)
kx2 = -1*kx2[::-1]
kx  = (2.* np.pi/L)*np.concatenate((kx1,kx2))

#----- Parameters -----
delta  = 0.022
delta2 = delta**2
TB  = 1./np.pi		# breakdown time
TE  = 3.6*TB		# end time

#----- Change of Variables -----
def uhat2vhat(t,uhat):
    return np.exp( -1j * (kx**3) * delta2 * t) * uhat

def vhat2uhat(t,vhat):
    return np.exp(1j * (kx**3) * delta2 * t) * vhat

#----- Define RHS -----
def uhatprime(t, uhat):
    u = np.fft.ifft(uhat)
    return 1j * (kx**2) * delta2 * uhat - 0.5j * kx * np.fft.fft(u**2)

def vhatprime(t, vhat):
    u = np.fft.ifft(vhat2uhat(t,vhat))
    return  -0.5j * kx * uhat2vhat(t, np.fft.fft(u**2) )

#------ Initial conditions -----
u0      = np.cos(np.pi*x)
uhat0   = np.fft.fft(u0)

#------ Solving for ODE -----
t0 = 0; tf = TB;
TimeSpan = [t0, tf]
nt       = 1500
vhat0    = uhat2vhat(t0,uhat0)



#------ Animation -----
fig = plt.figure()
fig.suptitle('Solitary-wave Pulses Simulation (KdV)', fontsize=14)
ax1 = plt.subplot(111,xlim=(0.,L),ylim=(-1.,3))
ax1.tick_params(axis='y', pad=8)
ax1.grid(True)
line, = ax1.plot(x,u0)
plt.xlabel('NORMALZIED DISTANCE')

vhatstream = RK4Stream(vhatprime,[t0,tf],vhat0,nt)

time_text1 = ax1.text(0.02, 0.92, '', fontsize=12,transform=ax1.transAxes)
time_text2 = ax1.text(0.70, 0.95, '', fontsize=10,transform=ax1.transAxes)
time_text3 = ax1.text(0.70, 0.88, '', fontsize=12,transform=ax1.transAxes)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=300, metadata=dict(artist='Jundong Qiao'))

def animate(i):
    t,vhat = vhatstream.next()
    tshow  = t/TB
    u = np.fft.ifft(vhat2uhat(t,vhat))
    line.set_ydata(np.real(u))
    time_text1.set_text('t = %.2f $t_B$' % tshow)
    time_text2.set_text('KdV Equation:')
    time_text3.set_text('$u_t$$+$$uu_x$$+$$\delta^{2} u_{xxx}$$= 0$')
    return line, time_text1, time_text2, time_text3

anim = animation.FuncAnimation(fig, animate, interval=15/nt+10, blit=False, save_count=5400)
anim.save('KdV.mp4', dpi = 300, writer=writer)

plt.show()
