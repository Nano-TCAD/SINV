import sympy as sy


# Define the symbol for Upper banded U matrix
k, n = sy.symbols('k,n')
zeros = sy.ZeroMatrix(n,n)


G00, G01, G02, G03 = sy.MatrixSymbol('G_{00}', n, n), sy.MatrixSymbol('G_{01}', n, n), sy.MatrixSymbol('G_{02}', n, n), sy.MatrixSymbol('G_{03}', n, n)
G10, G11, G12, G13 = sy.MatrixSymbol('G_{10}', n, n), sy.MatrixSymbol('G_{11}', n, n), sy.MatrixSymbol('G_{12}', n, n), sy.MatrixSymbol('G_{13}', n, n)
G20, G21, G22, G23 = sy.MatrixSymbol('G_{20}', n, n), sy.MatrixSymbol('G_{21}', n, n), sy.MatrixSymbol('G_{22}', n, n), sy.MatrixSymbol('G_{23}', n, n)
G30, G31, G32, G33 = sy.MatrixSymbol('G_{30}', n, n), sy.MatrixSymbol('G_{31}', n, n), sy.MatrixSymbol('G_{32}', n, n), sy.MatrixSymbol('G_{33}', n, n)

G = sy.Matrix([[G00, G01, G02, G03],
               [G10, G11, G12, G13],
               [G20, G21, G22, G23],
               [G30, G31, G32, G33]])


A00, A01, A10, A11, A12, A21, A22, A23, A32, A33 = sy.MatrixSymbol('A_{00}', n, n), sy.MatrixSymbol('A_{01}', n, n), sy.MatrixSymbol('A_{10}', n, n), sy.MatrixSymbol('A_{11}', n, n), sy.MatrixSymbol('A_{12}', n, n), sy.MatrixSymbol('A_{21}', n, n), sy.MatrixSymbol('A_{22}', n, n), sy.MatrixSymbol('A_{23}', n, n), sy.MatrixSymbol('A_{32}', n, n), sy.MatrixSymbol('A_{33}', n, n)

A = sy.Matrix([[A00,   A01,   zeros, zeros],
               [A10,   A11,   A12,   zeros],
               [zeros, A21,   A22,   A23],
               [zeros, zeros, A32,   A33]])


Sigma00, Sigma01, Sigma10, Sigma11, Sigma12, Sigma21, Sigma22, Sigma23, Sigma32, Sigma33 = sy.MatrixSymbol('\Sigma_{00}', n, n), sy.MatrixSymbol('\Sigma_{01}', n, n), sy.MatrixSymbol('\Sigma_{10}', n, n), sy.MatrixSymbol('\Sigma_{11}', n, n), sy.MatrixSymbol('\Sigma_{12}', n, n), sy.MatrixSymbol('\Sigma_{21}', n, n), sy.MatrixSymbol('\Sigma_{22}', n, n), sy.MatrixSymbol('\Sigma_{23}', n, n), sy.MatrixSymbol('\Sigma_{32}', n, n), sy.MatrixSymbol('\Sigma_{33}', n, n)

Sigma = sy.Matrix([[Sigma00, Sigma01, zeros,   zeros],
                    [Sigma10, Sigma11, Sigma12, zeros],
                    [zeros,       Sigma21, Sigma22, Sigma23],
                    [zeros,       zeros,       Sigma32, Sigma33]])


LHS = A @ G @ A.adjoint()

print(sy.latex(LHS))
sy.preview(LHS, viewer='file', filename='LHS.png', euler=False)