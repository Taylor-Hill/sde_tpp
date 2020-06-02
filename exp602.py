import jax
import jax.numpy as jnp
import numpy as onp
from jax import jacfwd, jacrev
from jax import grad, jit, vmap
from jax import random
from jax.experimental import stax
from jax import jit, grad, random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

nx = 2
tw = jnp.array([[0.5, 0.0], [0.0, 0.5]])
tbeta = random.normal(random.PRNGKey(0), (nx, 1))
x0 = onp.zeros(nx)
bs = 20

class Pa(nn.Module):
    def __init__(self):
        super(Pa, self).__init__()
        self.w = torch.nn.Parameter(torch.Tensor(onp.zeros_like(tw)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(onp.zeros_like(tbeta)), requires_grad=True)


def dxdtfunc(x, t, params):
    ax = jnp.matmul(params, x)
    # ax = jax.nn.sigmoid(ax)
    return ax

def dxfunc(xt, t, W, db):
    dx = jnp.zeros_like(xt)
    dxdt = dxdtfunc(xt, t, W)
    dx += db[0] * dxdt
    dx += db[1:]
    return dx

def getsample(x0, w, st=0.0, et=1.0, dt=0.01):
    nx = len(x0)
    ts = onp.linspace(st, et, int(et / dt)+1)[:-1]
    var = onp.sqrt(0.5 * dt)
    xt = x0

    for t in ts:
        db = onp.random.normal(0, var, nx)
        db = onp.insert(db, 0, dt)
        dx = dxfunc(xt, t, w, db)
        xt += dx

    return xt


def dxdufunc(y, t, params):
    return params.reshape(-1)

def daxdtfunc(x, t, p1, p2, y):
    dxdt = dxdtfunc(x, t, p1)
    dxdu = dxdufunc(y, t, p2)
    daxdt = dxdt + dxdu

    return daxdt

def daxfunc(xt, t, p1, p2, y, db):
    dx = jnp.zeros_like(xt)
    daxdt = daxdtfunc(xt, t, p1, p2, y)
    dx += db[0] * daxdt
    dx += db[1:]
    return dx


dxdpdtfunc_c = jax.jacfwd(daxdtfunc,(0, 2, 3))


def dxdpdtfunc(xt, y, p1, p2, t, dxdp1, dxdp2, db):
    gx, gp1, gp2 = dxdpdtfunc_c(xt, t, p1, p2, y)
    nx = len(xt)

    s = dxdp1.shape
    ggp1 = jnp.matmul(gx, dxdp1.reshape([nx, -1])).reshape(s)
    ggp1 += gp1

    s = dxdp2.shape
    ggp2 = jnp.matmul(gx, dxdp2.reshape([nx, -1])).reshape(s)
    ggp2 += gp2

    ggp1 *= db[0]
    ggp2 *= db[0]

    return ggp1, ggp2

def augx_dynamic_system(x0, p1, p2, st, et, y, dt = 0.05, dxonly=True):
    ts = onp.linspace(st, et, int(et/dt)+1)[:-1]
    xt = x0

    dxdp1t = jnp.zeros(x0.shape + p1.shape)
    dxdp2t = jnp.zeros(x0.shape + p2.shape)

    xt_list = [xt]
    for t in ts:
        dwv = onp.random.normal(0, onp.sqrt(0.5 * dt), nx)
        dwv = onp.insert(dwv, 0, dt)
        if dxonly is False:
            step_dxdp1t, step_dxdp2t = dxdpdtfunc(xt, y, p1, p2, t, dxdp1t, dxdp2t, dwv)
            dxdp1t += step_dxdp1t
            dxdp2t += step_dxdp2t

        dxt = daxfunc(xt, t, p1, p2, y, dwv)
        xt += dxt
        xt_list.append(xt)

    return xt, dxdp1t, dxdp2t

def likehood(x, mx):
    nx = len(x)
    ddx = x.reshape(-1) - mx.reshape(-1)
    coe = onp.float_power((2.0*onp.pi), -nx/2.0)
    return jnp.dot(ddx, ddx)/2.0

gradlikehood = jax.grad(likehood)

def get_batch_sample(bs = 10):
    x_list = []
    y_list = []
    for i in range(bs):
        x = getsample(x0, tw, st=0.0, et=1.0, dt=0.01)
        y = onp.random.multivariate_normal(x.reshape(-1), onp.identity(nx))
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list

model = Pa()

l = torch.sum(model.w) + torch.sum(model.beta)
l.backward()
model.zero_grad()



opter= optim.SGD(model.parameters(), lr=0.1)

def train(bs):

    fw = jnp.array([[0.8, 0.0],[0.0, 0.6]])
    fbeta = random.normal(random.PRNGKey(10), (nx, 1))

    model.w.data = torch.Tensor(onp.array(fw))
    model.beta.data = torch.Tensor(onp.array(fbeta))

    def ccalgrad(fw, fbeta, x_list, y_list, bs):

        gp1 = jnp.zeros_like(tw)
        gp2 = jnp.zeros_like(fbeta)
        lhs = 0.0

        sbs = 20
        for j in range(sbs):
            xpred, dxdp1t, dxdp2t = augx_dynamic_system(x0, fw, fbeta, 0.0, 1.0, y=0.0, dt=0.05, dxonly=False)
            for i in range(bs):
                y = y_list[i]
                lh = likehood(xpred, y)
                lhs += lh
                gygx = gradlikehood(xpred, y)
                s = dxdp1t.shape
                gygw = jnp.dot(gygx, dxdp1t.reshape([nx, -1])).reshape(s[1:])
                s = dxdp2t.shape
                gygbeta = jnp.dot(gygx, dxdp2t.reshape([nx, -1])).reshape(s[1:])
                gp1 += gygw
                gp2 += gygbeta
                gp2 += fbeta

        gp1 /= bs*sbs
        gp2 /= bs*sbs

        return gp1, gp2, lhs

    def calgrad(fw, fbeta, x_list, y_list, bs):

        gp1 = jnp.zeros_like(tw)
        gp2 = jnp.zeros_like(fbeta)
        lhs = 0.0

        for i in range(bs):
            x = x_list[i]
            y = y_list[i]
            lgp1 = jnp.zeros_like(tw)
            lgp2 = jnp.zeros_like(fbeta)
            for j in range(20):
                xpred, dxdp1t, dxdp2t = augx_dynamic_system(x0, fw, fbeta, 0.0, 1.0, y, dt = 0.05, dxonly=False)
                lh = likehood(xpred, y)
                lhs += lh
                gygx = gradlikehood(xpred, y)
                s = dxdp1t.shape
                gygw = jnp.dot(gygx, dxdp1t.reshape([nx, -1])).reshape(s[1:])
                s = dxdp2t.shape
                gygbeta = jnp.dot(gygx, dxdp2t.reshape([nx, -1])).reshape(s[1:])
                lgp1 += gygw
                lgp2 += gygbeta
                lgp2 += fbeta

            lgp1 /= 20.0
            lgp2 /= 20.0

            gp1 += lgp1
            gp2 += lgp2

        gp1 /= bs
        gp2 /= bs

        return gp1, gp2, lhs

    for i in range(200):
        x_list, y_list = get_batch_sample(bs)

        print(fbeta)
        print(fw)
        print(tw)

        opter.zero_grad()
        ind = i%bs
        gp1, gp2, lhs = ccalgrad(fw, fbeta, x_list, y_list, bs)
        print('----', i, lhs)

        # if i % 5 == 0:
        model.w.grad.data = torch.Tensor(onp.array(gp1))
        model.beta.grad.data = torch.Tensor(onp.array(gp2))
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        opter.step()
        fw = jnp.array(model.w.detach())
        fbeta = jnp.array(model.beta.detach())


train(bs)
