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

def metric_from_jacobian(fc,args,__time_zero = True):
    j = J(fc)
    # a = b = c
    # j.T is the transpose of j
    return simplify((j.T @ flat_metric(args,time_zero = __time_zero) @ j))

_sign_of_ads3 = (1,2)
_sign_of_ds3 = (2,1)
# %%
def scalar(metric_matrix, output=False):
    x = [u,v]
    m_obj = MetricTensor(metric_matrix, x)
    # print('metric'); display(m_obj);
    Ric = RicciTensor.from_metric(m_obj)
    print("Ricci tensor")
    display(Ric.tensor().simplify())
    R = RicciScalar.from_riccitensor(Ric)
    print("Ricci scalar")
    display(R.simplify())
    if output == True:
        return Ric, R

# %%
def main(g,space="ads"):
    x = [u,v]
    if space == "ads":
        print('anti-de Sitter space')
        anti_de_Sitter = g(one_sheet_hyperboloid, _sign_of_ads3)
        display(anti_de_Sitter)
        ################
        (scalar(anti_de_Sitter))
        ################
        
        m_obj = MetricTensor(anti_de_Sitter, x)
        ch = ChristoffelSymbols.from_metric(m_obj)
        display(ch.tensor())
    if space == "ds":
        print('de Sitter space = one-sheet hyperboloid')
        de_Sitter = g(one_sheet_hyperboloid, _sign_of_ds3,__time_zero = false)
        display(de_Sitter)
        ################
        Ric, R = (scalar(de_Sitter, output=True))
        Ric, R = Ric.simplify(), R.simplify()
        ################
        G = Ric - tensorproduct(R / 2,de_Sitter)
        display(G)
        
# %%
if __name__ == '__main__':
    print('one-sheet')
    display(J(one_sheet_hyperboloid))
    print('two-sheet')
    display(J(two_sheet_hyperboloid))

    main(metric_from_jacobian, space="ds")

# %%


# %%

# def main(g,f=two_sheet_hyperboloid):
#     print('one-sheet')
#     display(g(one_sheet_hyperboloid))
#     print('two-sheet')
#     display(g(two_sheet_hyperboloid))

#     x = [u,v]
#     m_obj = MetricTensor(g(f), x)
#     ch = ChristoffelSymbols.from_metric(m_obj)
#     display(ch.tensor())

# %%

