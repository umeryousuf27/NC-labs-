"""
Lab 1: Root Finding Methods
============================

This module implements various numerical methods for finding roots of equations f(x) = 0:
1. Bisection Method
2. False Position (Regula Falsi)
3. Fixed-Point Iteration
4. Newton-Raphson Method
5. Secant Method

Author: Numerical Methods Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict
import pandas as pd


class RootFindingMethods:
    """Collection of numerical root-finding algorithms."""
    
    @staticmethod
    def bisection(f: Callable, a: float, b: float, tol: float = 1e-6, 
                  max_iter: int = 100) -> Dict:
        """
        Bisection Method for finding roots.
        
        The bisection method repeatedly bisects an interval and selects a subinterval
        in which a root must lie for further processing.
        
        Parameters:
        -----------
        f : callable
            The function for which we want to find the root
        a : float
            Left endpoint of initial interval
        b : float
            Right endpoint of initial interval
        tol : float
            Tolerance for convergence (default: 1e-6)
        max_iter : int
            Maximum number of iterations (default: 100)
            
        Returns:
        --------
        dict : Dictionary containing:
            - root: The approximate root
            - iterations: Number of iterations performed
            - history: List of (iteration, a, b, c, f(c), error) tuples
            - converged: Boolean indicating if method converged
        """
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        history = []
        converged = False
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            error = abs(b - a) / 2
            
            history.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if error < tol or abs(fc) < tol:
                converged = True
                break
            
            if f(a) * fc < 0:
                b = c
            else:
                a = c
        
        return {
            'root': c,
            'iterations': i + 1,
            'history': history,
            'converged': converged
        }
    
    @staticmethod
    def false_position(f: Callable, a: float, b: float, tol: float = 1e-6,
                       max_iter: int = 100) -> Dict:
        """
        False Position (Regula Falsi) Method for finding roots.
        
        Similar to bisection but uses linear interpolation to find the next point
        instead of simply bisecting the interval.
        
        Parameters:
        -----------
        f : callable
            The function for which we want to find the root
        a : float
            Left endpoint of initial interval
        b : float
            Right endpoint of initial interval
        tol : float
            Tolerance for convergence (default: 1e-6)
        max_iter : int
            Maximum number of iterations (default: 100)
            
        Returns:
        --------
        dict : Dictionary containing root, iterations, history, and convergence status
        """
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        history = []
        converged = False
        c_old = a
        
        for i in range(max_iter):
            fa, fb = f(a), f(b)
            c = b - (fb * (b - a)) / (fb - fa)
            fc = f(c)
            error = abs(c - c_old) if i > 0 else abs(b - a)
            
            history.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if error < tol or abs(fc) < tol:
                converged = True
                break
            
            if f(a) * fc < 0:
                b = c
            else:
                a = c
            
            c_old = c
        
        return {
            'root': c,
            'iterations': i + 1,
            'history': history,
            'converged': converged
        }
    
    @staticmethod
    def fixed_point(g: Callable, x0: float, tol: float = 1e-6,
                    max_iter: int = 100) -> Dict:
        """
        Fixed-Point Iteration Method.
        
        Solves x = g(x) by iterating x_{n+1} = g(x_n).
        Note: Convergence requires |g'(x)| < 1 near the fixed point.
        
        Parameters:
        -----------
        g : callable
            The iteration function g(x) where we seek x = g(x)
        x0 : float
            Initial guess
        tol : float
            Tolerance for convergence (default: 1e-6)
        max_iter : int
            Maximum number of iterations (default: 100)
            
        Returns:
        --------
        dict : Dictionary containing root, iterations, history, and convergence status
        """
        history = []
        converged = False
        x = x0
        
        for i in range(max_iter):
            x_new = g(x)
            error = abs(x_new - x)
            
            history.append({
                'iteration': i + 1,
                'x': x,
                'g(x)': x_new,
                'error': error
            })
            
            if error < tol:
                converged = True
                x = x_new
                break
            
            x = x_new
        
        return {
            'root': x,
            'iterations': i + 1,
            'history': history,
            'converged': converged
        }
    
    @staticmethod
    def newton_raphson(f: Callable, df: Callable, x0: float, tol: float = 1e-6,
                       max_iter: int = 100) -> Dict:
        """
        Newton-Raphson Method for finding roots.
        
        Uses the formula: x_{n+1} = x_n - f(x_n) / f'(x_n)
        Requires both the function and its derivative.
        
        Parameters:
        -----------
        f : callable
            The function for which we want to find the root
        df : callable
            The derivative of f
        x0 : float
            Initial guess
        tol : float
            Tolerance for convergence (default: 1e-6)
        max_iter : int
            Maximum number of iterations (default: 100)
            
        Returns:
        --------
        dict : Dictionary containing root, iterations, history, and convergence status
        """
        history = []
        converged = False
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-12:
                raise ValueError(f"Derivative too small at iteration {i+1}")
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            history.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                "f'(x)": dfx,
                'x_new': x_new,
                'error': error
            })
            
            if error < tol or abs(fx) < tol:
                converged = True
                x = x_new
                break
            
            x = x_new
        
        return {
            'root': x,
            'iterations': i + 1,
            'history': history,
            'converged': converged
        }
    
    @staticmethod
    def secant(f: Callable, x0: float, x1: float, tol: float = 1e-6,
               max_iter: int = 100) -> Dict:
        """
        Secant Method for finding roots.
        
        Similar to Newton-Raphson but approximates the derivative using finite differences.
        Uses the formula: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        
        Parameters:
        -----------
        f : callable
            The function for which we want to find the root
        x0 : float
            First initial guess
        x1 : float
            Second initial guess
        tol : float
            Tolerance for convergence (default: 1e-6)
        max_iter : int
            Maximum number of iterations (default: 100)
            
        Returns:
        --------
        dict : Dictionary containing root, iterations, history, and convergence status
        """
        history = []
        converged = False
        
        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)
            
            if abs(fx1 - fx0) < 1e-12:
                raise ValueError(f"Division by zero at iteration {i+1}")
            
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            error = abs(x_new - x1)
            
            history.append({
                'iteration': i + 1,
                'x0': x0,
                'x1': x1,
                'f(x0)': fx0,
                'f(x1)': fx1,
                'x_new': x_new,
                'error': error
            })
            
            if error < tol or abs(fx1) < tol:
                converged = True
                x1 = x_new
                break
            
            x0, x1 = x1, x_new
        
        return {
            'root': x1,
            'iterations': i + 1,
            'history': history,
            'converged': converged
        }


def plot_convergence(results: Dict, method_name: str, save_path: str = None):
    """
    Plot convergence history for a root-finding method.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from a root-finding method
    method_name : str
        Name of the method for the plot title
    save_path : str, optional
        Path to save the figure
    """
    history = results['history']
    iterations = [h['iteration'] for h in history]
    errors = [h['error'] for h in history]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error (log scale)', fontsize=12)
    plt.title(f'{method_name} - Convergence Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_function_and_root(f: Callable, root: float, interval: Tuple[float, float],
                           method_name: str, save_path: str = None):
    """
    Plot the function and mark the found root.
    
    Parameters:
    -----------
    f : callable
        The function to plot
    root : float
        The root found by the method
    interval : tuple
        (a, b) interval to plot
    method_name : str
        Name of the method for the plot title
    save_path : str, optional
        Path to save the figure
    """
    x = np.linspace(interval[0], interval[1], 1000)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=root, color='r', linestyle='--', alpha=0.5, label=f'Root ≈ {root:.6f}')
    plt.plot(root, f(root), 'ro', markersize=12, label=f'f({root:.6f}) ≈ {f(root):.2e}')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(f'{method_name} - Function and Root', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_iteration_table(results: Dict, method_name: str):
    """
    Print a formatted table of iterations.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from a root-finding method
    method_name : str
        Name of the method for the table title
    """
    print(f"\n{'='*80}")
    print(f"{method_name} - Iteration History".center(80))
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(results['history'])
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"Final Result: Root = {results['root']:.10f}")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example usage with f(x) = x^3 - 2x - 5
    print("Lab 1: Root Finding Methods - Example Demonstrations")
    print("="*80)
    
    # Define test function
    f = lambda x: x**3 - 2*x - 5
    df = lambda x: 3*x**2 - 2  # Derivative for Newton-Raphson
    g = lambda x: (2*x + 5)**(1/3)  # Rearranged for fixed-point: x = (2x + 5)^(1/3)
    
    methods = RootFindingMethods()
    
    # 1. Bisection Method
    print("\n1. BISECTION METHOD")
    result_bisection = methods.bisection(f, 2, 3)
    print_iteration_table(result_bisection, "Bisection Method")
    
    # 2. False Position
    print("\n2. FALSE POSITION METHOD")
    result_false_pos = methods.false_position(f, 2, 3)
    print_iteration_table(result_false_pos, "False Position Method")
    
    # 3. Fixed-Point Iteration
    print("\n3. FIXED-POINT ITERATION")
    result_fixed = methods.fixed_point(g, 2.0)
    print_iteration_table(result_fixed, "Fixed-Point Iteration")
    
    # 4. Newton-Raphson
    print("\n4. NEWTON-RAPHSON METHOD")
    result_newton = methods.newton_raphson(f, df, 2.0)
    print_iteration_table(result_newton, "Newton-Raphson Method")
    
    # 5. Secant Method
    print("\n5. SECANT METHOD")
    result_secant = methods.secant(f, 2.0, 3.0)
    print_iteration_table(result_secant, "Secant Method")
