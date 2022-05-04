#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import matplotlib.pyplot as plt

plt.ion()
plt.close("all")
#

import gyptis as gy
import numpy as np

arch = np.load("contour.npz", allow_pickle=True)
contours = arch["contours"]
density_bin = arch["density_bin"]
# for contour in contours:
# geom.add_spline(pts)

a = 1
wavelength = a / 0.5670
nper = 4, 15
geom = gy.BoxPML(
    dim=2,
    box_size=(nper[0] + 3 * a, nper[1] * a + a),
    pml_width=(wavelength, wavelength),
)
box = geom.box

x = np.linspace(-nper[0] * a / 2, nper[0] * a / 2, nper[0] + 1)[:-1]
y = np.linspace(-nper[1] * a / 2, nper[1] * a / 2, nper[1] + 1)[:-1]
#
# downsple = 10
#
# objs=[]
# inside=[]
# for i in range(nper[0]):
#     for j in range(nper[1]):
#         # if i==1 and j==1:
#         #     break
#         print(i,j)
#         obj=[]
#
#         unitcell = geom.add_rectangle(x[i], y[j],0,a,a)
#         unitcell,box = geom.fragment(box,unitcell)
#         for contour in contours:
#             contour = contour[::downsple]
#             p = []
#             for c in contour:
#                 p.append(geom.add_point(c[0]+x[i], c[1]+y[j],0))
#             l=[]
#             for k,_ in enumerate( p[:-1]):
#                 try:
#                     l.append(geom.add_line(p[k],p[k+1]))
#                 except:
#                     pass
#             try:
#                 l.append(geom.add_line(p[-1],p[0]))
#             except:
#                 pass
#
#             ll = geom.add_curve_loop(l)
#             bit = geom.add_plane_surface([ll])
#
#
#
#             obj.append(bit)
#
#         # bit =  geom.add_rectangle(x[i]+a/4, y[j]+a/4,0,a/2,a/2)
#         # obj.append(bit)
#         out = geom.fragment(unitcell,obj)
#         unitcell = out[:len(obj)]
#         object = out[len(obj):]
#         objs.append(object)
#         inside.append(unitcell)
#
# inside = np.hstack(inside).tolist()
#
# *inside,box = geom.fragment(inside,box)
#
# objs = np.hstack(objs).tolist()
# # obj=np.array(obj).ravel()
# geom.add_physical(objs,"object")
# geom.add_physical(inside,"inside")
# geom.add_physical(box,"box")
# geom.set_pml_mesh_size(0.3)
# geom.set_size("box",0.1)
# geom.set_size("object",0.1)
# geom.set_size("inside",0.1)
# geom.build()
des = geom.add_rectangle(
    -nper[0] * a / 2, -nper[1] * a / 2, 0, nper[0] * a, nper[1] * a
)
des, box = geom.fragment(des, box)

geom.add_physical(des, "des")
geom.add_physical(box, "box")
geom.set_pml_mesh_size(0.3)
geom.set_size("box", 0.1)
geom.set_size("des", 0.05)
# geom.set_size("object",0.1)
# geom.set_size("inside",0.1)
geom.build()

# gb = gy.GaussianBeam(
#     wavelength=wavelength,
#     angle=gy.pi/2,
#     waist=wavelength,
#     position=(0, 0),
#     dim=2,
#     domain=geom.mesh,
#     degree=2,
# )

ls = gy.LineSource(
    wavelength=wavelength,
    position=(-nper[0] * a / 2 - 0.5 * a, 0),
    dim=2,
    domain=geom.mesh,
    degree=2,
)

# epsilon = dict(box=1, inside=1,object=9)
# mu = dict(box=1, inside=1,object=1)
epsilon = dict(box=1, des=1)
mu = dict(box=1, des=1)

#
# epsilon["des"]=9
#


V = gy.dolfin.FunctionSpace(geom.mesh, "DG", 0)

dens = gy.dolfin.Function(V)

dens_array = gy.utils.function2array(dens)

import scipy.interpolate as si

nx, ny = density_bin.shape

x = np.linspace(0, a, nx)
y = np.linspace(0, a, ny)
fi = si.RectBivariateSpline(x, y, density_bin)

i = 0
for x, y in V.tabulate_dof_coordinates():
    x = np.mod(x, a)
    y = np.mod(y, a)
    val = fi(x, y)
    val = 0 if val < 0.5 else 1
    dens_array[i] = val
    i += 1

dens = gy.utils.array2function(dens_array, V)
epsilon["des"] = (9 - 1) * dens + 1

s = gy.Scattering(
    geom,
    epsilon,
    mu,
    ls,
    degree=2,
    polarization="TE",
)
#
#
# testeps = s.formulation.epsilon
# testeps.plot(component=(0,0))
# gy.plot(dens)
# plt.axis("scaled")


s.solve()

plt.figure(figsize=(4, 1.8))
s.plot_field(type="real", cmap="RdBu_r")
# ax = plt.gca()
# for i in range(N):
#     cir = plt.Circle((-N / 2 * d + i * d, 0), R, lw=0.3, color="w", fill=False)
#     ax.add_patch(cir)
# plt.xlabel("x (nm)")
# plt.ylabel("y (nm)")
# plt.title(title)
plt.tight_layout()
