import sympy



# Define the symbol for Upper banded U matrix
U00, U11, U22, U33, U01, U12, U23 = sympy.Symbol('U_{00}'), sympy.Symbol('U_{11}'), sympy.Symbol('U_{22}'), sympy.Symbol('U_{33}'), sympy.Symbol('U_{01}'), sympy.Symbol('U_{12}'), sympy.Symbol('U_{23}')

U = sympy.Matrix([[U00, U01, 0,   0],
                  [0,   U11, U12, 0],
                  [0,   0,   U22, U23],
                  [0,   0,   0,   U33]])  

# Define the symbol for Upper banded U dagger matrix
U00H, U11H, U22H, U33H, U01H, U12H, U23H = sympy.Symbol('U_{00}^H'), sympy.Symbol('U_{11}^H'), sympy.Symbol('U_{22}^H'), sympy.Symbol('U_{33}^H'), sympy.Symbol('U_{01}^H'), sympy.Symbol('U_{12}^H'), sympy.Symbol('U_{23}^H')

UH = sympy.Matrix([[U00H, U01H, 0,    0],
                  [0,    U11H, U12H, 0],
                  [0,    0,    U22H, U23H],
                  [0,    0,    0,    U33H]]).transpose()

# Define the symbol for Lower banded L matrix
L00, L10, L11, L21, L22, L32, L33 = sympy.Symbol('L_{00}'), sympy.Symbol('L_{10}'), sympy.Symbol('L_{11}'), sympy.Symbol('L_{21}'), sympy.Symbol('L_{22}'), sympy.Symbol('L_{32}'), sympy.Symbol('L_{33}')

L = sympy.Matrix([[L00, 0,   0,     0],
                  [L10, L11, 0,     0],
                  [0,   L21, L22,   0],
                  [0,   0,   L32,   L33]])

# Define the symbol for Upper banded L dagger matrix
L00H, L10H, L11H, L21H, L22H, L32H, L33H = sympy.Symbol('L_{00}^H'), sympy.Symbol('L_{10}^H'), sympy.Symbol('L_{11}^H'), sympy.Symbol('L_{21}^H'), sympy.Symbol('L_{22}^H'), sympy.Symbol('L_{32}^H'), sympy.Symbol('L_{33}^H')

LH = sympy.Matrix([[L00H, 0,    0,     0],
                   [L10H, L11H, 0,     0],
                   [0,    L21H, L22H, 0],
                   [0,    0,    L32H, L33H]]).transpose()

# Define the symbol for G matrix
G00, G01, G10, G11, G12, G21, G22, G23, G32, G33 = sympy.Symbol('G_{00}'), sympy.Symbol('G_{01}'), sympy.Symbol('G_{10}'), sympy.Symbol('G_{11}'), sympy.Symbol('G_{12}'), sympy.Symbol('G_{21}'), sympy.Symbol('G_{22}'), sympy.Symbol('G_{23}'), sympy.Symbol('G_{32}'), sympy.Symbol('G_{33}')

G = sympy.Matrix([[G00, G01, 0,   0],
                  [G10, G11, G12, 0],
                  [0,   G21, G22, G23],
                  [0,   0,   G32, G33]])

# Define the symbol for Sigma matrix
Sigma00, Sigma01, Sigma10, Sigma11, Sigma12, Sigma21, Sigma22, Sigma23, Sigma32, Sigma33 = sympy.Symbol('\Sigma_{00}'), sympy.Symbol('\Sigma_{01}'), sympy.Symbol('\Sigma_{10}'), sympy.Symbol('\Sigma_{11}'), sympy.Symbol('\Sigma_{12}'), sympy.Symbol('\Sigma_{21}'), sympy.Symbol('\Sigma_{22}'), sympy.Symbol('\Sigma_{23}'), sympy.Symbol('\Sigma_{32}'), sympy.Symbol('\Sigma_{33}')

Sigma = sympy.Matrix([[Sigma00, Sigma01, 0,   0],
                      [Sigma10, Sigma11, Sigma12, 0],
                      [0,       Sigma21, Sigma22, Sigma23],
                      [0,       0,       Sigma32, Sigma33]])



LHS = L * U * G * U.transpose() * L.transpose()


""" # Define the equation
equation = sympy.Eq(L * U * G * U.transpose() * L.transpose(), Sigma)

# Solve the equation
solution = sympy.solve(equation, (G00, G01, G10, G11, G12, G21, G22, G23, G32, G33))

# Print the solution
print(solution) """


print(sympy.latex(LHS))
sympy.preview(LHS, viewer='file', filename='LHS.png', euler=False)



""" print(sympy.latex(U))
print(sympy.latex(L))
print(sympy.latex(G))
print(sympy.latex(Sigma)) 
print(sympy.latex(UH))  
print(sympy.latex(LH)) """