"""
Solving reaction-diffusion metabolic model in a Circular domain 
Reference Farina et al. DOI: https://doi.org/10.1101/2022.07.21.500921
geo:  https://doi.org/10.1038/s41598-020-71329-8

GLC influx subregion on the left, LAC outflux subregion on the right


"""


# from __future__ import print_function
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
# matplotlib.rcParams['text.usetex'] = True
#from matplotlib import rc
#rc('font',**{'family':'serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#from mshr import *
#from dolfin import *
from fenics import *
import numpy as np
from timeit import default_timer as timer
import sys
import meshio


#x_center,y_center = 138., 59.
x_center,y_center = 140., 70.
x_hxk, y_hxk = x_center,y_center
x_ldh, y_ldh = x_center,y_center
x_mito, y_mito = x_center,y_center
x_pyrk, y_pyrk = x_center,y_center


# Start timer
startime = timer()

T = 500 # final time
num_step = 2000 # number of time step
dt = T / num_step

## import mesh
msh = meshio.read("Astrocyte_mesh2Dfiner.msh")
meshio.write("mesh.xml",msh)
xml_file = Mesh("mesh.xml")
mesh = Mesh(xml_file)


## Create mesh
# N = 150
# radius = 70.
# channel = Circle(Point(36.05, 60), radius)
# mesh = generate_mesh(channel, N)

# Area 
#radius = 70.
#area = radius**2 * np.pi
area = 4862.22
# Finite Element space for the concentration
P1 = FiniteElement('P', mesh.ufl_cell(), 1)
element = MixedElement([P1,P1,P1,P1,P1,P1])
#element = VectorElement('P', mesh.ufl_cell(), 1, dim=6)
V = FunctionSpace(mesh,element) 

# Define test functions
v_1, v_2, v_3, v_4, v_5, v_6 = TestFunctions(V)

# Define Trial functions which must be Functions instead of Trial Functions cause the pb is non linear
u = Function (V)

# Define the initial condition of concetrations at t=0
a_0 = 0. # GLC
b_0 = 1.6 # ATP
c_0 = 1.6 # ADP
d_0 = 0.0 # GLY
e_0 = 0. # PYR
f_0 = 0. # LAC

u_0 = Expression(('a_0', 'b_0', 'c_0','d_0', 'e_0', 'f_0'), a_0=a_0, b_0=b_0, c_0=c_0, d_0=d_0, e_0=e_0, f_0=f_0, degree=1)
u_n = project(u_0, V)

a, b, c, d, e, f = split(u)
a_n, b_n, c_n, d_n, e_n, f_n = split(u_n)

# Time stepping
t = [0.0]

# Define Constant
k = Constant(dt)

# Define Reaction Rate
k_hxk = Constant(0.0619)
k_pyrk = Constant(1.92)
k_ldh = Constant(0.719)
k_mito = Constant(0.0813)
k_act = Constant(0.169)


# Variance of the Reaction rates
sigma = 20.0

# Colocalize d 2 and the other not
g_hxk = Expression("1./(pi*2*sigma*sigma) * exp(-((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))/(2*sigma*sigma))",
                   x0=x_hxk, y0=y_hxk, sigma=sigma, degree=2)

#g = interpolate(g_hxk, V)


#File("gaussian.pvd") << g

# Define the Gaussian function indicating where PYRK reaction take place
g_pyrk = Expression("1. /(pi*2*sigma*sigma) * exp(-((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))/(2*sigma*sigma))",
                    x0=x_pyrk, y0=y_pyrk, sigma=sigma, degree=2)


# Define the Gaussian function indicating where PYRK reaction take place
g_ldh = Expression("1. /(pi*2*sigma*sigma) * exp(-((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))/(2*sigma*sigma))",
                   x0=x_ldh, y0=y_ldh, sigma=sigma,degree=2)


# Define the Gaussian function indicating where PYRK reaction take place
g_mito = Expression("1. /(pi*2*sigma*sigma) * exp(-((x[0]-x0)*(x[0]-x0)+(x[1]-y0)*(x[1]-y0))/(2*sigma*sigma))",
                    x0=x_mito, y0=y_mito, sigma=sigma, degree=2)


# Adaptive Normalization

eta_hxk = assemble( g_hxk * dx(mesh))
eta_pyrk = assemble(g_pyrk*dx(mesh))
eta_ldh = assemble(g_ldh*dx(mesh))
eta_mito = assemble(g_mito*dx(mesh))


# Spatial reaction sites

K_hxk = g_hxk * k_hxk /Constant(eta_hxk) * Constant(area)
K_pyrk = g_pyrk * k_pyrk /Constant(eta_pyrk)* Constant(area)
K_ldh = g_ldh * k_ldh /Constant(eta_ldh)* Constant(area)
K_mito = g_mito * k_mito /Constant(eta_mito)* Constant(area)

# Classic reaction rate for cellular activity

K_act =Constant(k_act)

# Diffusion constant [\mu m^2/s]

D_a = Constant(0.6E3)
D_b = Constant(0.15E3)
D_c = Constant(0.15E3)
D_d = Constant(0.51E3)
D_e = Constant(0.64E3)
D_f = Constant(0.64E3)

# Define the subdomain for GLC entrance
radius_influx = 10.
subdomain = Expression('(pow(x[0]-77.1,2)+pow(x[1]-159.5,2)) < (r * r) ? 1. : 0', r=radius_influx, degree=1)
subdomain_area = assemble(subdomain * dx(mesh))

# Define influx of GLC in a subdomain of the circle

influx =  0.048 * area /subdomain_area
print('influx', influx)
f_1 = Expression('(pow(x[0]-77.1,2)+pow(x[1]-159.5,2)) < (r * r) ? influx : 0', influx=influx, r=radius_influx, degree=1)


# Degradation of LAC

q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})

# Define the subdomain for LAC exit

radius_outflux = 10.
subdomain_outflux = Expression('(pow(x[0]- 184.3,2)+pow(x[1]- 53.3,2)) < (r * r) ? 1. : 0', r=radius_outflux, degree=1)
subdomain_outflux_area = assemble(subdomain_outflux * dx(mesh))

outflux = 0.0969 * area / subdomain_outflux_area

eta_f = Expression('(pow(x[0] - 184.3,2)+pow(x[1]- 53.3,2)) < (r * r) ? outflux : 0', outflux = outflux, r=radius_outflux, degree=1)


# Weak form

F = ((a - a_n) / k) * v_1 * dx \
    + D_a * dot(grad(a), grad(v_1)) * dx + K_hxk * a * b**2 * v_1 * dx \
    + ((b - b_n) / k) * v_2 * dx  \
    + D_b * dot(grad(b), grad(v_2)) * dx + 2 * K_hxk * a * b**2 * v_2 * dx - 2 * K_pyrk * d *  c**2 * v_2 * dx - 28 * K_mito * e * c**28 * v_2 * dx + K_act * b * v_2 * dx\
    + ((c - c_n) / k)*v_3 * dx \
    + D_c * dot(grad(c), grad(v_3)) * dx - 2 * K_hxk * a * b**2 * v_3 * dx  + 2 * K_pyrk * d * c**2 * v_3 * dx - K_act * b * v_3 * dx + 28 * K_mito * e * c**28 * v_3 * dx\
    + ((d - d_n) / k)*v_4 * dx\
    + D_d * dot(grad(d),grad(v_4)) * dx - 2 * K_hxk * a * b**2 * v_4 * dx + K_pyrk * d * c**2 * v_4 * dx\
    + ((e - e_n) / k)*v_5 * dx\
    + D_e * dot(grad(e),grad(v_5)) * dx  - K_pyrk  * d * c**2 * v_5 * dx + K_ldh * e * v_5 * dx + K_mito * e * c**28 * v_5 * dx\
    + ((f - f_n) / k)*v_6 * dx\
    + D_f * dot(grad(f),grad(v_6)) * dx - K_ldh * e * v_6 * dx + eta_f * f * v_6 * dx\
    - f_1 * v_1 * dx

# Empty list to store the solution

list_a =[]
list_b =[]
list_c =[]
list_d =[]
list_e = []
list_f = []



# Add concentration values at t=0

list_a.append(assemble(a_n * dx)/area)
list_b.append(assemble(b_n * dx)/area)
list_c.append(assemble(c_n * dx)/area)
list_d.append(assemble(d_n * dx)/area)
list_e.append(assemble(e_n * dx)/area)
list_f.append(assemble(f_n * dx)/area)

# Store time
time_list = []
time_list.append(t[0])


Nmax = 50
toll_a = 1.e-5

for n in range(num_step):
    print(n)
    # Solve the variational form for time step
    solve( F == 0, u)

    # Save solution to file (VTK)
    _a, _b, _c, _d, _e, _f = u.split()


    # Update the previous solution
    u_n.assign(u)

    t[0] = t[0] + dt

    # Update time
    time_list.append(t[0])

    # Save the concentrations in a list

    list_a.append(assemble(_a * dx)/area)

    list_b.append(assemble(_b * dx)/area)

    list_c.append(assemble(_c * dx)/area)

    list_d.append(assemble(_d * dx)/area)

    list_e.append(assemble(_e * dx)/area)

    list_f.append(assemble(_f * dx)/area)

    saven= list(range(0,1001,10))
    saven[0]=1
    #if n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,100,200,400,600,800,1000]:
    if n in saven:
        #Create VTK files for visualization output
        vtkfile_glc = File('Real2D/glc/glc_2D_%sof1000s.pvd' % n)
        vtkfile_atp = File('Real2D/atp/atp_2D_%sof1000s.pvd' % n)
        vtkfile_adp = File('Real2D/adp/adp_2D_%sof1000s.pvd' % n)
        vtkfile_gly = File('Real2D/gly/gly_2D_%sof1000s.pvd' % n)
        vtkfile_pyr = File('Real2D/pyr/pyr_2D_%sof1000s.pvd' % n)
        vtkfile_lac = File('Real2D/lac/lac_2D_%sof1000s.pvd' % n)
        
        vtkfile_glc << (_a, t[0])
        vtkfile_atp << (_b, t[0])
        vtkfile_adp << (_c, t[0])
        vtkfile_gly << (_d, t[0])
        vtkfile_pyr << (_e, t[0])
        vtkfile_lac << (_f, t[0])



# Create a single list with all the solutions

list_of_list = [list_a, list_b, list_c, list_d, list_e, list_f, time_list]

# save using numpy

# stop time
aftersolve = timer()
tottime = aftersolve-startime

print('final time', tottime)
print(list_of_list)

np.save('./output_2D_Center', np.asarray(list_of_list))
