import sympy as sy


# Define the symbol for Upper banded U matrix
k, n = sy.symbols('k,n')
zeros = sy.ZeroMatrix(n,n)


G00, G01, G10, G11 = sy.MatrixSymbol('G_{00}', n, n), sy.MatrixSymbol('G_{01}', n, n), sy.MatrixSymbol('G_{10}', n, n), sy.MatrixSymbol('G_{11}', n, n)

G = sy.Matrix([[G00,   G01],
               [G10,   G11]])


Sigma00, Sigma01, Sigma10, Sigma11 = sy.MatrixSymbol('\Sigma_{00}', n, n), sy.MatrixSymbol('\Sigma_{01}', n, n), sy.MatrixSymbol('\Sigma_{10}', n, n), sy.MatrixSymbol('\Sigma_{11}', n, n)

Sigma = sy.Matrix([[Sigma00,   Sigma01],
               [Sigma10,   Sigma11]])


U00, U01, U11 = sy.MatrixSymbol('U_{00}', n, n), sy.MatrixSymbol('U_{01}', n, n), sy.MatrixSymbol('U_{11}', n, n)

U = sy.Matrix([[U00,   U01],
               [zeros,   U11]])



L00, L10, L11 = sy.MatrixSymbol('L_{00}', n, n), sy.MatrixSymbol('L_{10}', n, n), sy.MatrixSymbol('L_{11}', n, n)

L = sy.Matrix([[L00,   zeros],
               [L10,   L11]])

L_inv = sy.Matrix([[L00.inv(),   zeros],
                   [-L11.inv()*L10*L00.inv(),   L11.inv()]])

LHS = U * G * U.adjoint()
RHS = L_inv * Sigma * L_inv.adjoint()


print(sy.latex(LHS))
print(sy.latex(RHS))

