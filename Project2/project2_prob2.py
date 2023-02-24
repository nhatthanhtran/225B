import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
from tqdm.auto import trange

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

def implicit_midpoint(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,0] = initial

    for i in trange(len(t)-1):
        arr_results[:,i+1] = newton_krylov(lambda x: x - arr_results[:,i] -
            h*f(arr_results[:,i]/2 + x/2, t[i]), arr_results[:,i]).flatten()

    return arr_results

def trapezoid(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,0] = initial

    for i in trange(len(t)-1):
        arr_results[:,i+1] = newton_krylov(lambda x: x - arr_results[:,i] -
            0.5*h*(f(arr_results[:,i],t[i]) + f(x,t[i])), arr_results[:,i]).flatten()


    return arr_results

def bdf(f, t, h, initial):
    initial = initial.reshape(-1,1)
    t = t.reshape(-1,1)
    n, _ = initial.shape
    nt, _ = t.shape
    arr_results = np.zeros((n,nt))
    arr_results[:,:2] = explicit_euler_mod(f, t[:2], h, initial)

    for i in trange(len(t)-2):
        arr_results[:,i+2] = newton_krylov(lambda x: x - 4/3*arr_results[:,i+1] +
            1/3*arr_results[:,i] - 2/3*h*f(x,t[i+2]), arr_results[:,i+1]).flatten()


    return arr_results

def f(y,t): return y**2 - y**3 

delta = 1e-2
initial = np.array([delta])
h = [0.01]
a = 0
b = 2/delta
# t = np.arange(start=a, stop=b, step=h)

plot_type = ["whole", "zoom"]

lst_methods = [
    {"name": "Explicit Euler",
    "function": explicit_euler,
    "fig_name": "explicit_euler"},    
    {"name": "Implicit Midpoint",
    "function": implicit_midpoint,
    "fig_name": "implicit_mid"},
    {"name": "Trapezoid",
    "function": trapezoid,
    "fig_name": "trapezoid"},
    {"name": "bdf",
    "function": bdf,
    "fig_name": "bdf"},
]


for method in lst_methods:
    fig, axs = plt.subplots(nrows=len(h), ncols=len(plot_type), figsize=(16,9))

    for (step, pt), ax in zip(it.product(h,plot_type),axs.flat):
        t = np.arange(start=a, stop=b, step=step)
        s = method["function"](f, t, step, initial).flatten()

        if pt == "whole":
            ax.plot(t, s)
            ax.set_xlabel("t")
            ax.set_ylabel("y(t)")
            ax.set_title(f'Numerical Solution')
        elif pt == "zoom":
            ax.plot(t[:],s[:])
            ax.set_xlabel("t")
            ax.set_ylabel("y(t)")
            ax.set_ylim([1-1e-12,1+1e-12])
            ax.set_title(f'Numerical Solution Near Equilibrium')    
        # ax.plot(s[1,:])

    fig.tight_layout()
    fig.suptitle(method["name"])
    fig.tight_layout()
    plt.savefig("project2_prob2_" + method["fig_name"] + ".eps")
    plt.close