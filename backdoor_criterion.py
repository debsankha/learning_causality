# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The backdoor criterion

# %% [markdown]
# ## Define the data generation process

# %% tags=[]
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %matplotlib ipympl
matplotlib.rcParams["figure.figsize"] = (12, 4)
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["axes.grid"] = False

# %% [markdown]
# ```
#      U
#     x  x
#    x    x
#   x      x
#  x        x
# X------Z---Y 
# ```

# %% tags=[]
import numpy as np
from numpy.random import RandomState

K=3

def x(u, rs: RandomState):
    u = np.atleast_1d(u)
    return (u + rs.choice([-1,0,1], size=u.shape, p=[0.1, 0.7, 0.2]))

def z(x:int, rs: RandomState):
    x = np.atleast_1d(x)
    return (x + rs.choice([-1,0,1], size=x.shape, p=[0.2, 0.6, 0.2]))

def y(z:int, u:int, rs:RandomState):
    z, u = np.atleast_1d(z), np.atleast_1d(u)
    return (z+u+rs.choice([-1,0,1], size=u.shape, p=[0.1, 0.7, 0.2]))


# %% [markdown]
# ## Generate data

# %% tags=[]
N = 100000

U = np.random.RandomState(seed=0).randint(low=0, high=K, size=N)
X = x(U, RandomState(seed=1))
Z = z(X, RandomState(seed=2))
Y = y(Z, U, RandomState(seed=3))

# %% [markdown]
# ## Visualize data

# %% tags=[]
fig, axes = plt.subplots(ncols=4, figsize=(12, 4), sharey=True)

for var, ax in zip((U,X,Z,Y), axes):
    ax.hist(var, bins=np.arange(var.min(), var.max()+2), density=True)

# %% [markdown]
# ## Compute $P(Y|do(x=x'))$ by generating new data

# %% [markdown]
# ### Generate new data

# %% tags=[]
xp = 1

Up = np.random.RandomState(seed=0).randint(low=0, high=K, size=N)
Xp= np.ones_like(Up)*xp
Zp = z(Xp, RandomState(seed=2))
Yp = y(Zp, Up, RandomState(seed=3))

# %% tags=[]
fig, axes = plt.subplots(ncols=4, figsize=(12, 4))

for label, var, ax in zip(list("UXZY"), (Up,Xp,Zp,Yp), axes):
    ax.hist(var, bins=range(var.min(), var.max()+2), density=True)
    ax.set_title(label)
    
fig.tight_layout()

# %% [markdown]
# ## Compute $P(Y|do(x=x'))$ using the backdoor criterion
#
# $$
# P(y|do(x=x^*))=\sum_zP(z|x=x^*)\sum_{x'}P(y|x=x',z)P(x=x')
# $$

# %% tags=[]
est_p_z_given_x_is_xp = kde(dataset=Z[X==xp])

# %% tags=[]
from tqdm import tqdm

# %% tags=[]
np.logical_and(X==1, Y==3).sum()


# %% tags=[]
class EmptyDataError(Exception):
    ...

def backdoor_p_y_do_x(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, x:int):
    x_domain = np.arange(X.min(), X.max()+1)
    y_domain = np.arange(Y.min(), Y.max()+1)
    z_domain = np.arange(Z.min(), Z.max()+1)
    
    
    est_p_z_given_x_is_x = lambda z:np.logical_and(Z==z, X==x).sum()/(X==x).sum()
    est_p_x = lambda x_: (X==x_).sum()/len(X)
    
    # now apply the formula
    est_pmf = np.zeros_like(y_domain, dtype='float')
    for idx, y in enumerate(y_domain):
        p = 0
        for z in z_domain:
            p1 = est_p_z_given_x_is_x(z)
            p2 = 0
            for xp in x_domain:
                #est_p_y_given_x_and_z
                p2 += (np.logical_and(Y==y, X==xp, Z==z).sum()/np.logical_and(X==xp, Z==z).sum()) * est_p_x(xp)
            p+=p1*p2
        est_pmf[idx] = p
    return np.array(est_pmf)

# %% tags=[]
X.min(), X.max()

# %% tags=[]
pp = backdoor_p_y_do_x(X,Y,Z,xp)

# %%
pp

# %% [markdown]
# ## Does the backdoor criterion work?

# %% tags=[]
fig, ax = plt.subplots()

y_domain = np.arange(Y.min(), Y.max()+1)

ax.hist(Yp, bins=range(Yp.min(), Yp.max()+2), density=True, label='actual')
ax.plot(y_domain, pp, label='backdoor')

ax.legend()
fig.tight_layout()

# %% tags=[]
Y.min(), Y.max()

# %% tags=[]
pp.sum()

# %%
