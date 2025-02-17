import numpy as np
from src.main.assignment_2 import (
    neville_interpolation, newton_forward_coefficients, newton_forward_interpolation,
    create_hermite_table, cubic_spline_setup, solve_cubic_spline
)

def test_assignment_2():
    x_vals_neville = np.array([3.6, 3.8, 3.9])
    y_vals_neville = np.array([1.675, 1.436, 1.318])
    x_target_neville = 3.7
    result_neville = neville_interpolation(x_vals_neville, y_vals_neville, x_target_neville)
    print("#1")
    print(result_neville)
    print()

    x_vals_newton = np.array([7.2, 7.4, 7.5, 7.6])
    y_vals_newton = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    coeffs_newton = newton_forward_coefficients(x_vals_newton, y_vals_newton)
    x_target_newton = 7.3
    result_newton = newton_forward_interpolation(x_vals_newton, coeffs_newton, x_target_newton)
    print("#2")
    print(coeffs_newton[1:]) 
    print()
    print("#3")
    print(result_newton)
    print()
    
    x_vals_hermite = np.array([3.6, 3.8, 3.9])
    y_vals_hermite = np.array([1.675, 1.436, 1.318])
    dydx_vals_hermite = np.array([-1.195, -1.188, -1.182])
    hermite_table = create_hermite_table(x_vals_hermite, y_vals_hermite, dydx_vals_hermite)
    print("#4")
    for row in hermite_table:
        print([float(value) for value in row[:len(row) - 2]])  # Remove last two columns
    print()

    x_vals_spline = np.array([2, 5, 8, 10])
    y_vals_spline = np.array([3, 5, 7, 9])
    A_matrix, b_vector = cubic_spline_setup(x_vals_spline, y_vals_spline)
    print("#5a")
    print(A_matrix)
    print()

    print("#5b")
    print(b_vector)
    print()

    s_vector = solve_cubic_spline(A_matrix, b_vector)
    print("#5c")
    print(s_vector)
    print()

if __name__ == "__main__":
    test_assignment_2()
