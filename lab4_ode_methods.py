"""
LAB 4: DIFFERENTIAL EQUATIONS (ODE SOLVERS)
============================================

This module implements numerical methods for solving ordinary differential equations (ODEs)
of the form: dy/dx = f(x, y) with initial condition y(x₀) = y₀

Methods implemented:
- Euler's Method
- Midpoint Method
- Modified Euler Method
- Heun's Method
- 4th-Order Runge-Kutta (RK4)

Author: Numerical Methods Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class ODEResult:
    """Store results from ODE solver."""
    x_values: np.ndarray
    y_values: np.ndarray
    method_name: str
    step_size: float
    n_steps: int


class ODESolvers:
    """
    A comprehensive class for solving ODEs using various numerical methods.
    
    All methods solve: dy/dx = f(x, y) with y(x₀) = y₀
    """
    
    def __init__(self):
        """Initialize the ODE solvers class."""
        pass
    
    # ==================== EULER'S METHOD ====================
    
    def euler_method(self, f: Callable, x0: float, y0: float, 
                    x_end: float, h: float) -> ODEResult:
        """
        Euler's Method (First-order Runge-Kutta).
        
        The simplest numerical method for ODEs.
        
        Formula:
        --------
        y_{n+1} = y_n + h * f(x_n, y_n)
        
        Algorithm:
        ----------
        1. Start with (x₀, y₀)
        2. Compute slope: k = f(x_n, y_n)
        3. Update: y_{n+1} = y_n + h*k
        4. Update: x_{n+1} = x_n + h
        5. Repeat until x_end
        
        Characteristics:
        ----------------
        - Order: 1 (local error ~ O(h²), global error ~ O(h))
        - Simple but least accurate
        - Uses only one function evaluation per step
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
            
        Returns:
        --------
        ODEResult
            Contains x_values and y_values arrays
        """
        n_steps = int((x_end - x0) / h)
        x_values = np.zeros(n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        
        x_values[0] = x0
        y_values[0] = y0
        
        for i in range(n_steps):
            x_values[i + 1] = x_values[i] + h
            y_values[i + 1] = y_values[i] + h * f(x_values[i], y_values[i])
        
        return ODEResult(
            x_values=x_values,
            y_values=y_values,
            method_name="Euler's Method",
            step_size=h,
            n_steps=n_steps
        )
    
    # ==================== MIDPOINT METHOD ====================
    
    def midpoint_method(self, f: Callable, x0: float, y0: float,
                       x_end: float, h: float) -> ODEResult:
        """
        Midpoint Method (Second-order Runge-Kutta).
        
        Also known as the Modified Euler or RK2 method.
        
        Formula:
        --------
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h*k1/2)
        y_{n+1} = y_n + h * k2
        
        Algorithm:
        ----------
        1. Compute slope at beginning: k1 = f(x_n, y_n)
        2. Estimate midpoint: y_mid = y_n + (h/2)*k1
        3. Compute slope at midpoint: k2 = f(x_n + h/2, y_mid)
        4. Update using midpoint slope: y_{n+1} = y_n + h*k2
        
        Characteristics:
        ----------------
        - Order: 2 (local error ~ O(h³), global error ~ O(h²))
        - More accurate than Euler
        - Uses two function evaluations per step
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
            
        Returns:
        --------
        ODEResult
            Contains x_values and y_values arrays
        """
        n_steps = int((x_end - x0) / h)
        x_values = np.zeros(n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        
        x_values[0] = x0
        y_values[0] = y0
        
        for i in range(n_steps):
            k1 = f(x_values[i], y_values[i])
            k2 = f(x_values[i] + h/2, y_values[i] + h*k1/2)
            
            x_values[i + 1] = x_values[i] + h
            y_values[i + 1] = y_values[i] + h * k2
        
        return ODEResult(
            x_values=x_values,
            y_values=y_values,
            method_name="Midpoint Method",
            step_size=h,
            n_steps=n_steps
        )
    
    # ==================== MODIFIED EULER METHOD ====================
    
    def modified_euler_method(self, f: Callable, x0: float, y0: float,
                             x_end: float, h: float) -> ODEResult:
        """
        Modified Euler Method (Heun's Method without iteration).
        
        Also called the Improved Euler or Predictor-Corrector method.
        
        Formula:
        --------
        k1 = f(x_n, y_n)                    [predictor]
        k2 = f(x_n + h, y_n + h*k1)         [corrector]
        y_{n+1} = y_n + h * (k1 + k2) / 2
        
        Algorithm:
        ----------
        1. Predict using Euler: y*_{n+1} = y_n + h*f(x_n, y_n)
        2. Evaluate at predicted point: k2 = f(x_{n+1}, y*_{n+1})
        3. Correct using average slope: y_{n+1} = y_n + h*(k1 + k2)/2
        
        Characteristics:
        ----------------
        - Order: 2 (local error ~ O(h³), global error ~ O(h²))
        - Predictor-corrector approach
        - Two function evaluations per step
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
            
        Returns:
        --------
        ODEResult
            Contains x_values and y_values arrays
        """
        n_steps = int((x_end - x0) / h)
        x_values = np.zeros(n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        
        x_values[0] = x0
        y_values[0] = y0
        
        for i in range(n_steps):
            k1 = f(x_values[i], y_values[i])
            k2 = f(x_values[i] + h, y_values[i] + h*k1)
            
            x_values[i + 1] = x_values[i] + h
            y_values[i + 1] = y_values[i] + h * (k1 + k2) / 2
        
        return ODEResult(
            x_values=x_values,
            y_values=y_values,
            method_name="Modified Euler Method",
            step_size=h,
            n_steps=n_steps
        )
    
    # ==================== HEUN'S METHOD ====================
    
    def heun_method(self, f: Callable, x0: float, y0: float,
                   x_end: float, h: float, max_iter: int = 3) -> ODEResult:
        """
        Heun's Method (with iteration).
        
        An iterative predictor-corrector method.
        
        Formula:
        --------
        Predictor: y*_{n+1} = y_n + h*f(x_n, y_n)
        Corrector (iterate): y_{n+1}^(k+1) = y_n + h*[f(x_n,y_n) + f(x_{n+1},y_{n+1}^(k))]/2
        
        Algorithm:
        ----------
        1. Predict: y⁰_{n+1} = y_n + h*f(x_n, y_n)
        2. Iterate corrector until convergence or max iterations
        3. Final value is the corrected y_{n+1}
        
        Characteristics:
        ----------------
        - Order: 2 (can be higher with more iterations)
        - More accurate than Modified Euler
        - Multiple function evaluations per step
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
        max_iter : int
            Maximum corrector iterations
            
        Returns:
        --------
        ODEResult
            Contains x_values and y_values arrays
        """
        n_steps = int((x_end - x0) / h)
        x_values = np.zeros(n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        
        x_values[0] = x0
        y_values[0] = y0
        
        for i in range(n_steps):
            # Predictor
            y_pred = y_values[i] + h * f(x_values[i], y_values[i])
            
            # Corrector (iterate)
            y_corr = y_pred
            for _ in range(max_iter):
                k1 = f(x_values[i], y_values[i])
                k2 = f(x_values[i] + h, y_corr)
                y_corr = y_values[i] + h * (k1 + k2) / 2
            
            x_values[i + 1] = x_values[i] + h
            y_values[i + 1] = y_corr
        
        return ODEResult(
            x_values=x_values,
            y_values=y_values,
            method_name="Heun's Method",
            step_size=h,
            n_steps=n_steps
        )
    
    # ==================== RUNGE-KUTTA 4TH ORDER ====================
    
    def runge_kutta_4(self, f: Callable, x0: float, y0: float,
                     x_end: float, h: float) -> ODEResult:
        """
        4th-Order Runge-Kutta Method (RK4).
        
        The most widely used ODE solver. Excellent balance of accuracy and efficiency.
        
        Formula:
        --------
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h*k1/2)
        k3 = f(x_n + h/2, y_n + h*k2/2)
        k4 = f(x_n + h, y_n + h*k3)
        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        Algorithm:
        ----------
        1. Compute slope at beginning: k1
        2. Compute slope at midpoint using k1: k2
        3. Compute slope at midpoint using k2: k3
        4. Compute slope at end using k3: k4
        5. Weighted average: y_{n+1} = y_n + h*(k1 + 2k2 + 2k3 + k4)/6
        
        Characteristics:
        ----------------
        - Order: 4 (local error ~ O(h⁵), global error ~ O(h⁴))
        - Very accurate
        - Four function evaluations per step
        - Industry standard for many applications
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
            
        Returns:
        --------
        ODEResult
            Contains x_values and y_values arrays
        """
        n_steps = int((x_end - x0) / h)
        x_values = np.zeros(n_steps + 1)
        y_values = np.zeros(n_steps + 1)
        
        x_values[0] = x0
        y_values[0] = y0
        
        for i in range(n_steps):
            k1 = f(x_values[i], y_values[i])
            k2 = f(x_values[i] + h/2, y_values[i] + h*k1/2)
            k3 = f(x_values[i] + h/2, y_values[i] + h*k2/2)
            k4 = f(x_values[i] + h, y_values[i] + h*k3)
            
            x_values[i + 1] = x_values[i] + h
            y_values[i + 1] = y_values[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return ODEResult(
            x_values=x_values,
            y_values=y_values,
            method_name="Runge-Kutta 4th Order",
            step_size=h,
            n_steps=n_steps
        )
    
    # ==================== COMPARISON AND ANALYSIS ====================
    
    def compare_all_methods(self, f: Callable, x0: float, y0: float,
                           x_end: float, h: float,
                           exact_solution: Callable = None) -> Dict:
        """
        Compare all ODE methods on the same problem.
        
        Parameters:
        -----------
        f : Callable
            Function f(x, y) representing dy/dx
        x0 : float
            Initial x value
        y0 : float
            Initial y value
        x_end : float
            Final x value
        h : float
            Step size
        exact_solution : Callable, optional
            Function y_exact(x) for error calculation
            
        Returns:
        --------
        Dict
            Results from all methods
        """
        methods = {
            'Euler': self.euler_method,
            'Midpoint': self.midpoint_method,
            'Modified Euler': self.modified_euler_method,
            'Heun': self.heun_method,
            'RK4': self.runge_kutta_4
        }
        
        results = {}
        
        for name, method in methods.items():
            result = method(f, x0, y0, x_end, h)
            
            if exact_solution is not None:
                exact_values = np.array([exact_solution(x) for x in result.x_values])
                errors = np.abs(result.y_values - exact_values)
                max_error = np.max(errors)
                final_error = errors[-1]
            else:
                exact_values = None
                errors = None
                max_error = None
                final_error = None
            
            results[name] = {
                'result': result,
                'exact_values': exact_values,
                'errors': errors,
                'max_error': max_error,
                'final_error': final_error
            }
        
        return results
    
    def analyze_step_size_effect(self, f: Callable, x0: float, y0: float,
                                 x_end: float, h_values: List[float],
                                 exact_solution: Callable,
                                 method_name: str = 'RK4') -> Dict:
        """
        Analyze how step size affects accuracy.
        
        Parameters:
        -----------
        f : Callable
            ODE function
        x0, y0 : float
            Initial conditions
        x_end : float
            Final x value
        h_values : List[float]
            List of step sizes to test
        exact_solution : Callable
            Exact solution function
        method_name : str
            Which method to use
            
        Returns:
        --------
        Dict
            Error data for different step sizes
        """
        method_map = {
            'Euler': self.euler_method,
            'Midpoint': self.midpoint_method,
            'Modified Euler': self.modified_euler_method,
            'Heun': self.heun_method,
            'RK4': self.runge_kutta_4
        }
        
        method = method_map[method_name]
        
        analysis = {
            'h_values': h_values,
            'max_errors': [],
            'final_errors': []
        }
        
        for h in h_values:
            result = method(f, x0, y0, x_end, h)
            exact_values = np.array([exact_solution(x) for x in result.x_values])
            errors = np.abs(result.y_values - exact_values)
            
            analysis['max_errors'].append(np.max(errors))
            analysis['final_errors'].append(errors[-1])
        
        return analysis


# ==================== UTILITY FUNCTIONS ====================

def print_solution_table(result: ODEResult, exact_solution: Callable = None,
                        show_every: int = 1):
    """
    Print a formatted table of the ODE solution.
    
    Parameters:
    -----------
    result : ODEResult
        Solution from an ODE method
    exact_solution : Callable, optional
        Exact solution for comparison
    show_every : int
        Show every nth row (for large datasets)
    """
    print("\n" + "="*80)
    print(f"{result.method_name} - Solution Table")
    print(f"Step size h = {result.step_size}")
    print("="*80)
    
    if exact_solution is not None:
        print(f"{'i':>4} {'x':>12} {'y (approx)':>15} {'y (exact)':>15} {'Error':>15}")
        print("-"*80)
        
        for i in range(0, len(result.x_values), show_every):
            x = result.x_values[i]
            y_approx = result.y_values[i]
            y_exact = exact_solution(x)
            error = abs(y_approx - y_exact)
            
            print(f"{i:>4} {x:>12.6f} {y_approx:>15.10f} {y_exact:>15.10f} {error:>15.2e}")
    else:
        print(f"{'i':>4} {'x':>12} {'y':>15}")
        print("-"*80)
        
        for i in range(0, len(result.x_values), show_every):
            print(f"{i:>4} {result.x_values[i]:>12.6f} {result.y_values[i]:>15.10f}")
    
    print("="*80)


def print_comparison_table(comparison_results: Dict):
    """Print comparison of all methods."""
    print("\n" + "="*80)
    print("COMPARISON OF ALL ODE METHODS")
    print("="*80)
    print(f"{'Method':<20} {'Final y':>15} {'Max Error':>15} {'Final Error':>15}")
    print("-"*80)
    
    for name, data in comparison_results.items():
        final_y = data['result'].y_values[-1]
        max_err = data['max_error'] if data['max_error'] is not None else float('nan')
        final_err = data['final_error'] if data['final_error'] is not None else float('nan')
        
        print(f"{name:<20} {final_y:>15.10f} {max_err:>15.2e} {final_err:>15.2e}")
    
    print("="*80)


def plot_ode_solutions(comparison_results: Dict, title: str = "ODE Solutions Comparison"):
    """
    Plot all ODE solutions on the same graph.
    
    Parameters:
    -----------
    comparison_results : Dict
        Results from compare_all_methods
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot each method
    for name, data in comparison_results.items():
        result = data['result']
        plt.plot(result.x_values, result.y_values, '-o', 
                label=name, linewidth=2, markersize=4, alpha=0.7)
    
    # Plot exact solution if available
    if list(comparison_results.values())[0]['exact_values'] is not None:
        first_result = list(comparison_results.values())[0]['result']
        exact_vals = list(comparison_results.values())[0]['exact_values']
        plt.plot(first_result.x_values, exact_vals, 'k--', 
                label='Exact', linewidth=2, alpha=0.9)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_comparison(comparison_results: Dict, title: str = "Error Comparison"):
    """Plot errors for all methods."""
    plt.figure(figsize=(12, 6))
    
    for name, data in comparison_results.items():
        if data['errors'] is not None:
            result = data['result']
            plt.semilogy(result.x_values, data['errors'], '-o',
                        label=name, linewidth=2, markersize=4, alpha=0.7)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Absolute Error (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_step_size_analysis(analysis: Dict, method_name: str):
    """Plot error vs step size."""
    plt.figure(figsize=(10, 6))
    
    plt.loglog(analysis['h_values'], analysis['max_errors'], 'o-',
              label='Max Error', linewidth=2, markersize=8)
    plt.loglog(analysis['h_values'], analysis['final_errors'], 's-',
              label='Final Error', linewidth=2, markersize=8)
    
    plt.xlabel('Step Size (h)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title(f'Step Size Effect - {method_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


# ==================== DEMONSTRATION ====================

def demonstrate_lab4():
    """
    Comprehensive demonstration of Lab 4: ODE Solvers.
    """
    print("\n" + "="*80)
    print("LAB 4: DIFFERENTIAL EQUATIONS (ODE SOLVERS) - DEMONSTRATION")
    print("="*80)
    
    # Test problem: dy/dx = y - x², y(0) = 1
    # Exact solution: y = x² + 2x + 2 - e^x
    f = lambda x, y: y - x**2
    exact = lambda x: x**2 + 2*x + 2 - np.exp(x)
    
    x0, y0 = 0, 1
    x_end = 1.0
    h = 0.1
    
    print(f"\nTest Problem: dy/dx = y - x²")
    print(f"Initial Condition: y({x0}) = {y0}")
    print(f"Interval: [{x0}, {x_end}]")
    print(f"Step size: h = {h}")
    print(f"Exact solution: y = x² + 2x + 2 - e^x")
    
    # Initialize solver
    solver = ODESolvers()
    
    # Compare all methods
    print("\n" + "-"*80)
    print("COMPARING ALL METHODS")
    print("-"*80)
    
    results = solver.compare_all_methods(f, x0, y0, x_end, h, exact)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print detailed table for RK4
    print("\n" + "-"*80)
    print("DETAILED SOLUTION TABLE (RK4)")
    print("-"*80)
    print_solution_table(results['RK4']['result'], exact, show_every=1)
    
    # Plot solutions
    print("\nGenerating plots...")
    plot_ode_solutions(results, "ODE Solutions: dy/dx = y - x²")
    plot_error_comparison(results, "Error Comparison: All Methods")
    
    # Step size analysis
    print("\n" + "-"*80)
    print("STEP SIZE ANALYSIS (RK4)")
    print("-"*80)
    
    h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
    analysis = solver.analyze_step_size_effect(f, x0, y0, x_end, h_values, exact, 'RK4')
    
    print(f"{'h':>10} {'Max Error':>15} {'Final Error':>15}")
    print("-"*80)
    for i, h_val in enumerate(h_values):
        print(f"{h_val:>10.4f} {analysis['max_errors'][i]:>15.2e} {analysis['final_errors'][i]:>15.2e}")
    
    plot_step_size_analysis(analysis, "Runge-Kutta 4th Order")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demonstrate_lab4()
