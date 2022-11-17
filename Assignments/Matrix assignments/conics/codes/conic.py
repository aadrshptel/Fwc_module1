import sys
import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
#from line.funcs import *
#sys.path.insert(0,'/home/root1/Downloads/CoordGeo') 
from numpy import linalg as LA
#from conics.funcs import *
#`from line.funcs import *
#from scipy.integrate import quad
import subprocess
import shlex
#  plotting parabola
x = np.linspace(-7, 7, 1000)
y = ((x ** 2)/4) 
y = np.linspace(-7, 7, 1000)
x = ((y ** 2)/4)
#x = np.linspace(-10, 10, 1000)
#y = x 
plt.plot(x, y, label='Parabola 1')
plt.plot(y, x, label='Parabola 2')
#plt.plot(x, y, label='Parabola')
def conic_intersect(V1,u1,f1,V2,u2,f2):
    
    K1 = np.block([[V1,u1],[u1.T,f1]])
    K2 = np.block([[V2,u2],[u2.T,f2]])
    
    x = sp.Symbol('x')
    M1 = sp.Matrix(K1)
    M2 = sp.Matrix(K2)
    M = M1 + x*M2
    eq = M.det()
    soln = sp.solveset(eq, x)
    print(soln)

    intersect_pts = []
    
    for i in soln:
        if(not i.is_real):
            print("complex found")
            continue
        
        mu = float(i)
        K = K1 + mu*K2
        V = K[0:-1,0:-1]
        u = K[0:-1, -1].reshape(2,1)
        f = K[-1,-1]

        lamda, gamma = LA.eigh(V)
        if(lamda[1] == 0):      # If eigen value negative, present at start of lamda 
            lamda = np.flip(lamda)
            gamma = np.flip(gamma,axis=1)
        
        if(LA.det(V) > 0):
            print("Det is positive")
            continue
        elif(LA.det(V) == 0):
            p1 = gamma[:,0].reshape(2,1)
            eta = u.T@p1
            print("eta:",eta)
            a = np.vstack((u.T + eta*p1.T, V))
            print(a)
            b = np.vstack((-f, eta*p1-u)) 
            c = LA.lstsq(a,b,rcond=None)[0]
            print("Det is 0", c)
        else:
            print("Det is negative")
            c = -LA.inv(V)@u

        n1 = gamma@np.array([np.sqrt(np.abs(lamda[0])), np.sqrt(np.abs(lamda[1]))])
        n1 = n1.reshape(2,1)
        n2 = gamma@np.array([np.sqrt(np.abs(lamda[0])), -np.sqrt(np.abs(lamda[1]))])
        n2 = n2.reshape(2,1)

        omat = np.array([[0,-1],[1,0]])
        m1 = omat@n1
        m2 = omat@n2
        
        print("vector m1",end=" ")
        code, p1, p2 = line_conic_intersect(V1,u1,f1,c,m1)
        if(code == 0):
            print("1")
            intersect_pts.append(p1)
        elif(code > 0):
            print("2")
            intersect_pts.append(p1)
            intersect_pts.append(p2)
        else:
            print("0")
            pass

        print("vector m2",end=" ")
        code, p1, p2 = line_conic_intersect(V1,u1,f1,c,m2)
        if(code == 0):
            print("1")
            intersect_pts.append(p1)
        elif(code > 0):
            print("2")
            intersect_pts.append(p1)
            intersect_pts.append(p2)
        else:
            print("0")
            pass
    
    return np.unique(intersect_pts, axis=0) 
#For first parabola
m=np.array([1,1]);#direction vector
q= np.array([0,0]);
V1=np.array([[1,0],[0,0]]);
u1=np.array([0,-2]);
f=0;
d1 = np.sqrt((m.T@(V1@q + u1)**2) - (q.T@V1@q + 2*u1.T@q + f)*(m.T@V1@m))
print("d is =",d1)
k1 = (d1 - m.T@(V1@q + u1))/(m.T@V1@m)
k2 = (-d1 - m.T@(V1@q + u1))/(m.T@V1@m)
print("k1 =",k1)
print("k2 =",k2)
a0 = q + k1*m
a1 = q + k2*m
#p1,p2=inter_pt(m,q,V,u,f)
print("a0 =",a0)
print("a1 =",a1)

#for second parabola
V2=np.array([[0,0],[0,1]]);
u2=np.array([-2,0]);
f=0;
d2 = np.sqrt((m.T@(V2@q + u2)**2) - (q.T@V2@q + 2*u2.T@q + f)*(m.T@V2@m))
print("d2 is =",d2)
k3 = (d2 - m.T@(V2@q + u2))/(m.T@V2@m)
k4 = (-d2 - m.T@(V2@q + u2))/(m.T@V2@m)
print("k3 =",k3)
print("k4 =",k4)
a2 = q + k3*m
a3 = q + k4*m
#p1,p2=inter_pt(m,q,V,u,f)
print("a2 =",a2)
print("a3 =",a3)


tri_coords = np.vstack((a1,a0)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a1(0,0)','a0(4a,4a)']
for i, txt in enumerate(vert_labels):
      plt.annotate(txt,      # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points",   # how to position the text
                 xytext=(0,10),     # distance from text to points (x,y)
                 ha='center')     # horizontal alignm'''

plt.axis('equal')
plt.legend(loc='best')
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.plot(x, x+0, linestyle='solid')
plt.grid()
plt.savefig('/home/adarsh/fiig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/adarsh/matrix/conics/fig.pdf"))
plt.show()
