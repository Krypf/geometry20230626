# %%
from sympy import *
from sympy.abc import a,b,c,u
v = Symbol('varphi') # v instead of phi

from IPython.core.display import display

from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor,RicciTensor, RicciScalar,EinsteinTensor

# %%
def one_sheet_hyperboloid(u,v):
    x = a * cosh(u) * cos(v)
    y = b * cosh(u) * sin(v)
    z = c * sinh(u)
    return Matrix([x,y,z]).subs(b,a).subs(c,a)
def two_sheet_hyperboloid(u,v):
    x = a * sinh(u) * cos(v)
    y = b * sinh(u) * sin(v)
    z = c * cosh(u)
    return Matrix([x,y,z]).subs(b,a).subs(c,a)
# %%
def J(f,args=(u,v)):# jacobian matrix
    x1, x2 = args[0], args[1]
    X = Matrix([x1,x2])
    return f(x1,x2).jacobian(X)

def jacobian_matrix(fc,x):# jacobian matrix
    return fc(x).jacobian(Matrix(x))

# surface definition end
# %%
def eta(n):# Minkowski metric
    x = eye(n)
    x[-1,-1] = -1
    return x

def flat_metric(args,time_zero = True):
    r, s = args[0], args[1]
    # the number of plus one = r
    # the number of minus one = s
    n = r + s
    x = eye(n)
    
    for j in range(s):
        if time_zero:
            k = j
        else:
            k = -(j+1)
        
        x[k,k] = - 1
    return x
# flat_metric(1,2) = diag(-1,-1,1)
# %%

def g1(f):# metric
    j = J(f)
    return simplify((j.T @ j))

def metric_from_jacobian(fc,x,args,__time_zero = True):
    jac = jacobian_matrix(fc,x)
    # jac.T is the transpose of j
    return simplify((jac.T @ flat_metric(args,time_zero = __time_zero) @ jac))

_sign_of_ads3 = (1,2)
_sign_of_ds3 = (2,1)
# %%
def scalar(metric_matrix, x, output=True):
    m_obj = MetricTensor(metric_matrix, x)
    # print('metric'); display(m_obj);
    Ric = RicciTensor.from_metric(m_obj)
    R = RicciScalar.from_riccitensor(Ric)
    
    if output == True:
        print("Ricci tensor")
        display(Ric.tensor().simplify())
        
        print("Ricci scalar")
        display(R.simplify())
    
    return Ric, R

def einstein_tensor(metric, x, output=True):
    ################
    Ric, R = (scalar(metric, x, output=True))
    Ric, R = Ric.simplify(), R.simplify()
    ################
    G = Ric - tensorproduct(R / 2,metric)
    G = G.simplify()
    if output == True:
        print("Einstein tensor")
        display(G)
    
    return (G)
# %%
import sympy as sp

def spherical_to_cartesian(n,radius = sp.Symbol("r"),subscript=0):
    
    # Define symbolic variables
    # angle
    
    if subscript == 0:
        spherical_vars = sp.symbols('theta0:%d' % (n-1))
    if subscript == 1:
        spherical_vars = sp.symbols('theta1:%d' % (n))
    
    # Create the coordinate mappings
    cartesian_coords = []
    polar_angle = spherical_vars[0]# theta_0
    cartesian_coords.append(radius * sp.cos(polar_angle))
    product_sin = sp.sin(polar_angle)
    
    for j in range(1, n - 1):
        cartesian_coords.append(radius * sp.cos(spherical_vars[j]))
        cartesian_coords[j] *= product_sin
        product_sin *= sp.sin(spherical_vars[j])
    
    cartesian_coords.append(radius * product_sin)

    print(cartesian_coords)
    
    return sp.Matrix(cartesian_coords)

# %%
def spherical_to_de_Sitter(n,radius = sp.Symbol("a")):
    # Define symbolic variables
    cartesian_coords_sphere = spherical_to_cartesian(n,radius=radius,subscript=1)
    u = sp.Symbol("u")
    cartesian_coords_sphere *= sp.cosh(u)
    cartesian_coords_sphere = list(cartesian_coords_sphere)
    # add the last coordinate
    cartesian_coords_sphere.append(radius * sp.sinh(u))
    return sp.Matrix(cartesian_coords_sphere)


