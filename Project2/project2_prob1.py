import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov

def explicit_euler_mod(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    for i in range(len(t)-1):
        k1 = f(arr_results[:,i],t[i])
        k2 = f(arr_results[:,i] + h*k1,t[i]+h)
        arr_results[:,i+1] = arr_results[:,i] + h*(0.5*k1 + 0.5*k2).flatten()
    return arr_results

def rk2(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    for i in range(len(t)-1):
        k1 = h*f(arr_results[:,i],t[i])
        k2 = h*f(arr_results[:,i] + k1/2,t[i]+h/2)
        arr_results[:,i+1] = arr_results[:,i] + k2.flatten()
    return arr_results 

def rk4(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    for i in range(len(t)-1):
        k1 = f(arr_results[:,i],t[i])
        k2 = f(arr_results[:,i] + h*k1/2, t[i]+h/2)
        k3 = f(arr_results[:,i] + h*k2/2, t[i]+h/2)
        k4 = f(arr_results[:,i] + h*k3,t[i]+h)
        k2 = h*f(arr_results[:,i] + k1/2,t[i]+h/2)
        arr_results[:,i+1] = arr_results[:,i] + h*1/6*(k1 + 2*k2 + 2*k3 +k4).flatten()
    return arr_results 

def explicit_euler(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    for i in range(len(t)-1):
        arr_results[:,i+1] = arr_results[:,i] + h*f(arr_results[:,i].flatten(),t[i]).flatten()

    return arr_results

def symplectic_euler(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    for i in range(len(t)-1):
        # compute q
        arr_results[1,i+1] = arr_results[1,i] + h*arr_results[0,i]
        # compute p
        arr_results[0,i+1] = arr_results[0,i] - h*np.sin(arr_results[1,i+1])

    return arr_results

def implicit_midpoint(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n, nt))
    arr_results[:,0] = initial.flatten()
    k2 = np.zeros((2,1))   
    for i in range(len(t)-1):
        k1 = f(arr_results[:,i],t[i])
        # k2[0,0] = newton_krylov(lambda x: k2_eq(x, arr_results[1,i], h),0)
        # k2[1,0] = 2*arr_results[0,i]/h 
        k2 = newton_krylov(lambda x: x - h*f(arr_results[:,i] + 0.5*x,t[i]+0.5*h),arr_results[:,i])
        arr_results[:,i+1] = arr_results[:,i] + k2.flatten()
    return arr_results

def multi_step_methodA(f, t, h, initial):

    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,:4] = initial

    for i in np.arange(4, len(t)):
        arr_results[:,i] = (h**2)*(7/6*f(arr_results[:,i-1], t[i-1]) - 
            5/12*f(arr_results[:,i-2], t[i-2]) + 1/3*f(arr_results[:,i-3], t[i-3]) -
            1/12*f(arr_results[:,i-4],t[i-4])) + (
                2*arr_results[:,i-1] - arr_results[:,i-2])

    return arr_results

def multi_step_methodB(f, t, h, initial):

    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,:4] = initial

    for i in np.arange(4, len(t)):
        arr_results[:,i] = (h**2)*(4/3*f(arr_results[:,i-1], t[i-1]) + 
            4/3*f(arr_results[:,i-2], t[i-2]) + 4/3*f(arr_results[:,i-3], t[i-3])) + (
            2*arr_results[:,i-2] - arr_results[:,i-4])

    return arr_results

def multi_step_methodC(f, t, h, initial):
    
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,:4] = initial

    for i in np.arange(4, len(t)):
        arr_results[:,i] = (h**2)*(7/6*f(arr_results[:,i-1], t[i-1]) - 
            1/3*f(arr_results[:,i-2], t[i-2]) + 7/6*f(arr_results[:,i-3], t[i-3])) + (
            2*arr_results[:,i-1] - 2*arr_results[:,i-2] + 
            2*arr_results[:,i-3] - arr_results[:,i-4])

    return arr_results

def adam_bashforth(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,:2] = explicit_euler_mod(f, t[:2], h, initial)

    for i in np.arange(2, len(t)):
        arr_results[:,i] = (h)*(3/2*f(arr_results[:,i-1], t[i-1]) - 
            1/2*f(arr_results[:,i-2], t[i-2])) + arr_results[:,i-1]

    return arr_results

def compute_energy(arr_p):
    return np.power(arr_p[0,:],2)/2 + 1- np.cos(arr_p[1,:]).flatten()

def f1(x, t): return np.array([-np.sin(x[1]), x[0]])
def f(x, t): return np.array([-np.cos(x[1])*x[0], -np.sin(x[1])])

a = 0
b = 60

initial1= np.array([0,2])
h = [0.08]

t = np.arange(start=a, stop=b, step=h[0])
initial = explicit_euler_mod(f1, t[:4], h[0], initial1)

# #use this to create a really good initial condtions
# initial[:,0] = initial1
# initial[:,1] = explicit_euler_mod(f1, np.arange(t[0],t[1]+h[0]/32,h[0]/32), h[0]/32, initial1)[:,-1]
# initial[:,2] = explicit_euler_mod(f1, np.arange(t[0],t[2]+h[0]/64,h[0]/64), h[0]/64, initial1)[:,-1]
# initial[:,3] = explicit_euler_mod(f1, np.arange(t[0],t[3]+h[0]/128,h[0]/128), h[0]/128, initial1)[:,-1]


plot_type = ["phase plane", "energy"]

lst_methods = [
    {"name": "method A",
    "function": multi_step_methodA,
    "fig_name": "methodA"},
    {"name": "method B",
    "function": multi_step_methodB,
    "fig_name": "methodB"},
    {"name": "method C",
    "function": multi_step_methodC,
    "fig_name": "methodC"},
] 

for method in lst_methods:
    fig, axs = plt.subplots(nrows=len(h), ncols=len(plot_type), figsize=(16,9))

    for (step, pt), ax in zip(it.product(h,plot_type),axs.flat):
        t = np.arange(start=a, stop=b, step=step)
        s = method["function"](f, t, step, initial)
        energy = compute_energy(s)
        if pt == "energy":
            ax.plot(t, energy)
            ax.set_xlabel("x")
            ax.set_ylabel("Energy")
            ax.set_title(f'Energy plot for h = {step}')
        else:
            ax.plot(s[1,:],s[0,:])
            ax.set_xlabel("q")
            ax.set_ylabel("p")
            ax.set_title(f'Phase plane plot for h = {step}')    
        # ax.plot(s[1,:])

    fig.tight_layout()
    fig.suptitle(method["name"])
    fig.tight_layout()
    plt.savefig("project2_prob1_" + method["fig_name"] + "best.eps")
    plt.close