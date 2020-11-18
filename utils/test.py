import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

xy = [[0.3,0.5],
      [0.6,0.8],
      [0.5,0.1],
      [0.1,0.2]]
xy = np.array(xy)

triangles = [[0,2,1],
             [2,0,3]]

triang = mtri.Triangulation(xy[:,0], xy[:,1], triangles=triangles)

z = [0.1,0.2,0.3,0.4]

fig, ax = plt.subplots(subplot_kw =dict(projection="3d"))
ax.plot_trisurf(triang, z)

plt.show()