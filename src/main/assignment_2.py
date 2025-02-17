
import numpy as np
from scipy.linalg import solve


def neville_interpolation(x_vals, y_vals, x):
    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = y_vals
    
    for i in range(1, n):
        for j in range(n - i):
            Q[j, i] = ((x - x_vals[j + i]) * Q[j, i - 1] - (x - x_vals[j]) * Q[j + 1, i - 1]) / (x_vals[j] - x_vals[j + i])
    
    return Q[0, n - 1]

def newton_forward_coefficients(x_vals, y_vals):
    n = len(x_vals)
    coef = np.zeros(n)
    coef[0] = y_vals[0]
    
    for i in range(1, n):
        y_vals = [(y_vals[j + 1] - y_vals[j]) / (x_vals[j + i] - x_vals[j]) for j in range(n - i)]
        coef[i] = y_vals[0]
    
    return coef

def newton_forward_interpolation(x_vals, coeffs, x):
    n = len(coeffs)
    approx = coeffs[0]
    product_term = 1
    
    for i in range(1, n):
        product_term *= (x - x_vals[i - 1])
        approx += coeffs[i] * product_term
    
    return approx

def create_hermite_table(x_points, fx_points, dfx_points):
    n = len(x_points)
    total_points = 2 * n
    
    # Initialize table with 0s
    table = [[0] * (total_points + 1) for _ in range(total_points)]
    
    # Fill z values and f(z)
    for i in range(n):
        table[2*i][0] = x_points[i]      # z value
        table[2*i+1][0] = x_points[i]    # z value (repeated)
        table[2*i][1] = fx_points[i]     # f(z)
        table[2*i+1][1] = fx_points[i]   # f(z) (repeated)
    
    # Fill derivatives in first d.d. column
    for i in range(n):
        table[2*i + 1][2] = dfx_points[i]
    
    # Row 3 (index 2): Calculate first and second d.d.
    table[2][2] = (table[2][1] - table[0][1]) / (table[2][0] - table[0][0])
    table[2][3] = (table[2][2] - table[1][2]) / (table[2][0] - table[0][0])
    
    # Row 4 (index 3): Gets first d.d. from derivatives
    table[3][3] = (table[3][2] - table[2][2]) / (table[3][0] - table[1][0])
    
    # Row 5 (index 4): Calculate first and second d.d.
    table[4][2] = (table[4][1] - table[2][1]) / (table[4][0] - table[2][0])
    table[4][3] = (table[4][2] - table[3][2]) / (table[4][0] - table[2][0])
    
    # Row 6 (index 5): Gets first d.d. from derivatives
    table[5][3] = (table[5][2] - table[4][2]) / (table[5][0] - table[3][0])
    
    # Third differences
    table[3][4] = (table[3][3] - table[2][3]) / (table[3][0] - table[0][0])
    table[4][4] = (table[4][3] - table[3][3]) / (table[4][0] - table[1][0])
    table[5][4] = (table[5][3] - table[4][3]) / (table[5][0] - table[2][0])
    
    return table

def cubic_spline_setup(x, f):
    n = len(x)

    # Construct the A matrix
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n-1, n-1] = 1

    for i in range(1, n-1):
        h_i = x[i] - x[i-1]
        h_i_plus_1 = x[i+1] - x[i]

        A[i, i-1] = h_i
        A[i, i] = 2 * (h_i + h_i_plus_1)
        A[i, i+1] = h_i_plus_1

    # Construct the b vector
    b = np.zeros(n)
    for i in range(1, n-1):
        h_i = x[i] - x[i-1]
        h_i_plus_1 = x[i+1] - x[i]
        b[i] = 3 * ((f[i+1] - f[i]) / h_i_plus_1 - (f[i] - f[i-1]) / h_i)

    return A, b


def solve_cubic_spline(A, b):
    s = solve(A, b)
    return s