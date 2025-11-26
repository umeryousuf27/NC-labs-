"""
Lab 2: Interpolation & Polynomial Approximation
================================================

This module implements various interpolation methods:
1. Lagrange Interpolation (degrees 1, 2, 3)
2. Newton Divided Difference
3. Newton Forward Difference
4. Newton Backward Difference

Author: Numerical Methods Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import pandas as pd


class InterpolationMethods:
    """Collection of polynomial interpolation algorithms."""
    
    @staticmethod
    def lagrange_interpolation(x_data: np.ndarray, y_data: np.ndarray) -> Callable:
        """
        Lagrange Interpolation Polynomial.
        
        Constructs a polynomial of degree n-1 that passes through n data points
        using the Lagrange basis polynomials.
        
        Formula: P(x) = Σ y_i * L_i(x)
        where L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        callable : Function that evaluates the Lagrange polynomial at any point
        """
        n = len(x_data)
        
        def lagrange_basis(i: int, x: float) -> float:
            """Compute the i-th Lagrange basis polynomial at x."""
            L = 1.0
            for j in range(n):
                if j != i:
                    L *= (x - x_data[j]) / (x_data[i] - x_data[j])
            return L
        
        def polynomial(x):
            """Evaluate the Lagrange polynomial at x."""
            if isinstance(x, np.ndarray):
                return np.array([polynomial(xi) for xi in x])
            
            result = 0.0
            for i in range(n):
                result += y_data[i] * lagrange_basis(i, x)
            return result
        
        return polynomial
    
    @staticmethod
    def lagrange_polynomial_string(x_data: np.ndarray, y_data: np.ndarray) -> str:
        """
        Generate a string representation of the Lagrange polynomial.
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        str : String representation of the polynomial
        """
        n = len(x_data)
        terms = []
        
        for i in range(n):
            # Build numerator and denominator for L_i(x)
            numerator_terms = []
            denominator = 1.0
            
            for j in range(n):
                if j != i:
                    numerator_terms.append(f"(x - {x_data[j]})")
                    denominator *= (x_data[i] - x_data[j])
            
            numerator = " * ".join(numerator_terms)
            coefficient = y_data[i] / denominator
            
            if numerator:
                terms.append(f"{coefficient:.4f} * {numerator}")
            else:
                terms.append(f"{coefficient:.4f}")
        
        return " + ".join(terms)
    
    @staticmethod
    def divided_difference_table(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """
        Compute the divided difference table for Newton's interpolation.
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        np.ndarray : Divided difference table (2D array)
        """
        n = len(x_data)
        table = np.zeros((n, n))
        table[:, 0] = y_data
        
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / \
                              (x_data[i + j] - x_data[i])
        
        return table
    
    @staticmethod
    def newton_divided_difference(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Callable, np.ndarray]:
        """
        Newton's Divided Difference Interpolation.
        
        Constructs the Newton polynomial using divided differences.
        Formula: P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        tuple : (polynomial function, divided difference table)
        """
        table = InterpolationMethods.divided_difference_table(x_data, y_data)
        n = len(x_data)
        
        def polynomial(x):
            """Evaluate the Newton polynomial at x."""
            if isinstance(x, np.ndarray):
                return np.array([polynomial(xi) for xi in x])
            
            result = table[0, 0]
            product = 1.0
            
            for i in range(1, n):
                product *= (x - x_data[i - 1])
                result += table[0, i] * product
            
            return result
        
        return polynomial, table
    
    @staticmethod
    def print_divided_difference_table(x_data: np.ndarray, y_data: np.ndarray):
        """
        Print the divided difference table in a formatted manner.
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of x coordinates
        y_data : np.ndarray
            Array of y coordinates
        """
        table = InterpolationMethods.divided_difference_table(x_data, y_data)
        n = len(x_data)
        
        print("\n" + "="*100)
        print("Newton's Divided Difference Table".center(100))
        print("="*100 + "\n")
        
        # Create header
        headers = ['x_i', 'f[x_i]']
        for i in range(1, n):
            if i == 1:
                headers.append('f[x_i, x_i+1]')
            elif i == 2:
                headers.append('f[x_i, x_i+1, x_i+2]')
            else:
                headers.append(f'f[...{i+1} points]')
        
        # Create data rows
        data = []
        for i in range(n):
            row = [x_data[i]]
            for j in range(n):
                if i + j < n:
                    row.append(table[i, j])
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        print(df.to_string(index=False, na_rep=''))
        print("\n" + "="*100 + "\n")
    
    @staticmethod
    def forward_difference_table(y_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward difference table.
        
        Parameters:
        -----------
        y_data : np.ndarray
            Array of y coordinates (equally spaced x values assumed)
            
        Returns:
        --------
        np.ndarray : Forward difference table
        """
        n = len(y_data)
        table = np.zeros((n, n))
        table[:, 0] = y_data
        
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
        
        return table
    
    @staticmethod
    def newton_forward(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Callable, np.ndarray]:
        """
        Newton's Forward Difference Interpolation.
        
        Used for interpolation near the beginning of tabulated data.
        Assumes equally spaced x values.
        
        Formula: P(x) = y_0 + u*Δy_0 + u(u-1)/2! * Δ²y_0 + ...
        where u = (x - x_0) / h
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of equally spaced x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        tuple : (polynomial function, forward difference table)
        """
        table = InterpolationMethods.forward_difference_table(y_data)
        n = len(x_data)
        h = x_data[1] - x_data[0]  # Step size
        x0 = x_data[0]
        
        def polynomial(x):
            """Evaluate the Newton forward polynomial at x."""
            if isinstance(x, np.ndarray):
                return np.array([polynomial(xi) for xi in x])
            
            u = (x - x0) / h
            result = table[0, 0]
            u_term = 1.0
            
            for i in range(1, n):
                u_term *= (u - i + 1) / i
                result += u_term * table[0, i]
            
            return result
        
        return polynomial, table
    
    @staticmethod
    def backward_difference_table(y_data: np.ndarray) -> np.ndarray:
        """
        Compute the backward difference table.
        
        Parameters:
        -----------
        y_data : np.ndarray
            Array of y coordinates (equally spaced x values assumed)
            
        Returns:
        --------
        np.ndarray : Backward difference table
        """
        n = len(y_data)
        table = np.zeros((n, n))
        table[:, 0] = y_data
        
        for j in range(1, n):
            for i in range(j, n):
                table[i, j] = table[i, j - 1] - table[i - 1, j - 1]
        
        return table
    
    @staticmethod
    def newton_backward(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[Callable, np.ndarray]:
        """
        Newton's Backward Difference Interpolation.
        
        Used for interpolation near the end of tabulated data.
        Assumes equally spaced x values.
        
        Formula: P(x) = y_n + v*∇y_n + v(v+1)/2! * ∇²y_n + ...
        where v = (x - x_n) / h
        
        Parameters:
        -----------
        x_data : np.ndarray
            Array of equally spaced x coordinates
        y_data : np.ndarray
            Array of y coordinates
            
        Returns:
        --------
        tuple : (polynomial function, backward difference table)
        """
        table = InterpolationMethods.backward_difference_table(y_data)
        n = len(x_data)
        h = x_data[1] - x_data[0]  # Step size
        xn = x_data[-1]
        
        def polynomial(x):
            """Evaluate the Newton backward polynomial at x."""
            if isinstance(x, np.ndarray):
                return np.array([polynomial(xi) for xi in x])
            
            v = (x - xn) / h
            result = table[n - 1, 0]
            v_term = 1.0
            
            for i in range(1, n):
                v_term *= (v + i - 1) / i
                result += v_term * table[n - 1, i]
            
            return result
        
        return polynomial, table
    
    @staticmethod
    def print_forward_difference_table(x_data: np.ndarray, y_data: np.ndarray):
        """Print the forward difference table."""
        table = InterpolationMethods.forward_difference_table(y_data)
        n = len(y_data)
        
        print("\n" + "="*100)
        print("Newton's Forward Difference Table".center(100))
        print("="*100 + "\n")
        
        headers = ['x_i', 'y_i'] + [f'Δ^{i}y' for i in range(1, n)]
        
        data = []
        for i in range(n):
            row = [x_data[i]]
            for j in range(n):
                if i + j < n:
                    row.append(table[i, j])
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        print(df.to_string(index=False, na_rep=''))
        print("\n" + "="*100 + "\n")
    
    @staticmethod
    def print_backward_difference_table(x_data: np.ndarray, y_data: np.ndarray):
        """Print the backward difference table."""
        table = InterpolationMethods.backward_difference_table(y_data)
        n = len(y_data)
        
        print("\n" + "="*100)
        print("Newton's Backward Difference Table".center(100))
        print("="*100 + "\n")
        
        headers = ['x_i', 'y_i'] + [f'∇^{i}y' for i in range(1, n)]
        
        data = []
        for i in range(n):
            row = [x_data[i]]
            for j in range(n):
                if i >= j:
                    row.append(table[i, j])
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        print(df.to_string(index=False, na_rep=''))
        print("\n" + "="*100 + "\n")


def plot_interpolation(x_data: np.ndarray, y_data: np.ndarray, 
                       polynomial: Callable, method_name: str,
                       x_range: Tuple[float, float] = None,
                       save_path: str = None):
    """
    Plot the interpolation polynomial along with data points.
    
    Parameters:
    -----------
    x_data : np.ndarray
        Original x data points
    y_data : np.ndarray
        Original y data points
    polynomial : callable
        The interpolation polynomial function
    method_name : str
        Name of the interpolation method
    x_range : tuple, optional
        (min, max) range for plotting
    save_path : str, optional
        Path to save the figure
    """
    if x_range is None:
        margin = (x_data[-1] - x_data[0]) * 0.1
        x_range = (x_data[0] - margin, x_data[-1] + margin)
    
    x_plot = np.linspace(x_range[0], x_range[1], 500)
    y_plot = polynomial(x_plot)
    
    plt.figure(figsize=(12, 7))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Interpolation Polynomial')
    plt.plot(x_data, y_data, 'ro', markersize=10, label='Data Points', zorder=5)
    
    # Add grid points labels
    for i, (xi, yi) in enumerate(zip(x_data, y_data)):
        plt.annotate(f'({xi:.2f}, {yi:.2f})', 
                    xy=(xi, yi), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'{method_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("Lab 2: Interpolation & Polynomial Approximation - Example Demonstrations")
    print("="*100)
    
    # Example 1: Lagrange Interpolation (different degrees)
    print("\n" + "="*100)
    print("EXAMPLE 1: LAGRANGE INTERPOLATION")
    print("="*100)
    
    # Linear (degree 1)
    print("\n--- Linear Interpolation (2 points) ---")
    x_linear = np.array([1.0, 3.0])
    y_linear = np.array([2.0, 8.0])
    poly_linear = InterpolationMethods.lagrange_interpolation(x_linear, y_linear)
    print(f"Data points: {list(zip(x_linear, y_linear))}")
    print(f"P(2.0) = {poly_linear(2.0):.4f}")
    
    # Quadratic (degree 2)
    print("\n--- Quadratic Interpolation (3 points) ---")
    x_quad = np.array([1.0, 2.0, 4.0])
    y_quad = np.array([1.0, 4.0, 2.0])
    poly_quad = InterpolationMethods.lagrange_interpolation(x_quad, y_quad)
    print(f"Data points: {list(zip(x_quad, y_quad))}")
    print(f"P(3.0) = {poly_quad(3.0):.4f}")
    
    # Cubic (degree 3)
    print("\n--- Cubic Interpolation (4 points) ---")
    x_cubic = np.array([0.0, 1.0, 2.0, 3.0])
    y_cubic = np.array([1.0, 2.0, 5.0, 10.0])
    poly_cubic = InterpolationMethods.lagrange_interpolation(x_cubic, y_cubic)
    print(f"Data points: {list(zip(x_cubic, y_cubic))}")
    print(f"P(1.5) = {poly_cubic(1.5):.4f}")
    
    # Example 2: Newton Divided Difference
    print("\n" + "="*100)
    print("EXAMPLE 2: NEWTON DIVIDED DIFFERENCE")
    print("="*100)
    
    x_newton = np.array([1.0, 1.5, 2.0, 2.5])
    y_newton = np.array([0.7651977, 0.8109302, 0.8451542, 0.8712010])
    
    poly_newton, table_newton = InterpolationMethods.newton_divided_difference(x_newton, y_newton)
    InterpolationMethods.print_divided_difference_table(x_newton, y_newton)
    
    print(f"Interpolated value at x=1.75: P(1.75) = {poly_newton(1.75):.7f}")
    
    # Example 3: Newton Forward Difference
    print("\n" + "="*100)
    print("EXAMPLE 3: NEWTON FORWARD DIFFERENCE")
    print("="*100)
    
    x_forward = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_forward = np.array([1.0, 1.6487, 2.7183, 4.4817, 7.3891])
    
    poly_forward, table_forward = InterpolationMethods.newton_forward(x_forward, y_forward)
    InterpolationMethods.print_forward_difference_table(x_forward, y_forward)
    
    print(f"Interpolated value at x=0.25: P(0.25) = {poly_forward(0.25):.4f}")
    
    # Example 4: Newton Backward Difference
    print("\n" + "="*100)
    print("EXAMPLE 4: NEWTON BACKWARD DIFFERENCE")
    print("="*100)
    
    x_backward = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_backward = np.array([1.0, 1.6487, 2.7183, 4.4817, 7.3891])
    
    poly_backward, table_backward = InterpolationMethods.newton_backward(x_backward, y_backward)
    InterpolationMethods.print_backward_difference_table(x_backward, y_backward)
    
    print(f"Interpolated value at x=1.75: P(1.75) = {poly_backward(1.75):.4f}")
