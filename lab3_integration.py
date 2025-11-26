"""
LAB 3: NUMERICAL INTEGRATION
=============================

This module implements various numerical integration techniques including:
- Basic integration rules (Trapezoidal, Simpson's 1/3, Simpson's 3/8)
- Closed Newton-Cotes formulas
- Composite integration methods

Author: Numerical Methods Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class IntegrationResult:
    """Store results from numerical integration."""
    approximation: float
    true_value: float
    error: float
    n_intervals: int
    method_name: str
    subinterval_values: List[float] = None


class NumericalIntegration:
    """
    A comprehensive class for numerical integration methods.
    
    Implements various techniques for approximating definite integrals.
    """
    
    def __init__(self):
        """Initialize the numerical integration class."""
        pass
    
    # ==================== BASIC INTEGRATION RULES ====================
    
    def trapezoidal_rule(self, f: Callable, a: float, b: float) -> float:
        """
        Single application of the Trapezoidal Rule.
        
        Formula: I ≈ (b-a)/2 * [f(a) + f(b)]
        
        Geometric interpretation: Approximates area under curve with a trapezoid.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit of integration
        b : float
            Upper limit of integration
            
        Returns:
        --------
        float
            Approximation of the integral
            
        Example:
        --------
        >>> ni = NumericalIntegration()
        >>> f = lambda x: x**2
        >>> result = ni.trapezoidal_rule(f, 0, 1)
        >>> print(f"Approximation: {result:.6f}")
        """
        h = b - a
        return (h / 2) * (f(a) + f(b))
    
    def simpsons_1_3_rule(self, f: Callable, a: float, b: float) -> float:
        """
        Single application of Simpson's 1/3 Rule.
        
        Formula: I ≈ (b-a)/6 * [f(a) + 4*f((a+b)/2) + f(b)]
        
        Uses a parabola (quadratic polynomial) to approximate the curve.
        More accurate than trapezoidal rule for smooth functions.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
            
        Returns:
        --------
        float
            Approximation of the integral
        """
        h = (b - a) / 2
        x_mid = (a + b) / 2
        return (h / 3) * (f(a) + 4 * f(x_mid) + f(b))
    
    def simpsons_3_8_rule(self, f: Callable, a: float, b: float) -> float:
        """
        Single application of Simpson's 3/8 Rule.
        
        Formula: I ≈ (b-a)/8 * [f(x0) + 3*f(x1) + 3*f(x2) + f(x3)]
        where x0=a, x1=a+h, x2=a+2h, x3=b, and h=(b-a)/3
        
        Uses a cubic polynomial. Useful when the interval needs to be
        divided into 3 segments.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
            
        Returns:
        --------
        float
            Approximation of the integral
        """
        h = (b - a) / 3
        x1 = a + h
        x2 = a + 2 * h
        return (3 * h / 8) * (f(a) + 3 * f(x1) + 3 * f(x2) + f(b))
    
    # ==================== NEWTON-COTES FORMULAS ====================
    
    def newton_cotes_degree_0(self, f: Callable, a: float, b: float) -> float:
        """
        Closed Newton-Cotes formula of degree 0 (Midpoint Rule).
        
        Formula: I ≈ (b-a) * f((a+b)/2)
        
        Approximates the area with a rectangle using the midpoint value.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
            
        Returns:
        --------
        float
            Approximation of the integral
        """
        return (b - a) * f((a + b) / 2)
    
    def newton_cotes_degree_1(self, f: Callable, a: float, b: float) -> float:
        """
        Closed Newton-Cotes formula of degree 1 (Trapezoidal Rule).
        
        Same as trapezoidal_rule. Included for completeness.
        """
        return self.trapezoidal_rule(f, a, b)
    
    def newton_cotes_degree_2(self, f: Callable, a: float, b: float) -> float:
        """
        Closed Newton-Cotes formula of degree 2 (Simpson's 1/3 Rule).
        
        Same as simpsons_1_3_rule. Included for completeness.
        """
        return self.simpsons_1_3_rule(f, a, b)
    
    # ==================== COMPOSITE INTEGRATION METHODS ====================
    
    def composite_trapezoidal(self, f: Callable, a: float, b: float, n: int) -> Tuple[float, List[float]]:
        """
        Composite Trapezoidal Rule.
        
        Divides [a,b] into n subintervals and applies trapezoidal rule to each.
        
        Formula: I ≈ h/2 * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(x_{n-1}) + f(xn)]
        where h = (b-a)/n
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
        n : int
            Number of subintervals
            
        Returns:
        --------
        Tuple[float, List[float]]
            (approximation, list of function values at each point)
        """
        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)
        y_values = [f(x) for x in x_values]
        
        # Apply composite trapezoidal formula
        integral = h / 2 * (y_values[0] + 2 * sum(y_values[1:-1]) + y_values[-1])
        
        return integral, y_values
    
    def composite_simpsons_1_3(self, f: Callable, a: float, b: float, n: int) -> Tuple[float, List[float]]:
        """
        Composite Simpson's 1/3 Rule.
        
        Requires n to be EVEN. Divides [a,b] into n subintervals.
        
        Formula: I ≈ h/3 * [f(x0) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + f(xn)]
        where h = (b-a)/n
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
        n : int
            Number of subintervals (must be even)
            
        Returns:
        --------
        Tuple[float, List[float]]
            (approximation, list of function values)
            
        Raises:
        -------
        ValueError
            If n is not even
        """
        if n % 2 != 0:
            raise ValueError("n must be even for Simpson's 1/3 rule")
        
        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)
        y_values = [f(x) for x in x_values]
        
        # Apply composite Simpson's 1/3 formula
        integral = h / 3 * (
            y_values[0] +
            4 * sum(y_values[i] for i in range(1, n, 2)) +
            2 * sum(y_values[i] for i in range(2, n, 2)) +
            y_values[n]
        )
        
        return integral, y_values
    
    def composite_midpoint(self, f: Callable, a: float, b: float, n: int) -> Tuple[float, List[float]]:
        """
        Composite Midpoint Rule.
        
        Divides [a,b] into n subintervals and uses midpoint of each.
        
        Formula: I ≈ h * [f(x0.5) + f(x1.5) + ... + f(x_{n-0.5})]
        where h = (b-a)/n and xi.5 is the midpoint of the i-th interval
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
        n : int
            Number of subintervals
            
        Returns:
        --------
        Tuple[float, List[float]]
            (approximation, list of midpoint function values)
        """
        h = (b - a) / n
        midpoints = [a + (i + 0.5) * h for i in range(n)]
        y_values = [f(x) for x in midpoints]
        
        integral = h * sum(y_values)
        
        return integral, y_values
    
    # ==================== ANALYSIS AND COMPARISON ====================
    
    def compare_methods(self, f: Callable, a: float, b: float, 
                       true_value: float, n_values: List[int] = None) -> Dict:
        """
        Compare different integration methods.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
        true_value : float
            True value of the integral (for error calculation)
        n_values : List[int], optional
            List of n values to test for composite methods
            
        Returns:
        --------
        Dict
            Dictionary containing results for each method
        """
        if n_values is None:
            n_values = [2, 4, 8, 16, 32, 64]
        
        results = {
            'basic_methods': {},
            'composite_methods': {}
        }
        
        # Basic methods (single application)
        basic_methods = {
            'Trapezoidal': self.trapezoidal_rule,
            "Simpson's 1/3": self.simpsons_1_3_rule,
            "Simpson's 3/8": self.simpsons_3_8_rule,
            'Midpoint (NC-0)': self.newton_cotes_degree_0
        }
        
        for name, method in basic_methods.items():
            approx = method(f, a, b)
            error = abs(approx - true_value)
            results['basic_methods'][name] = {
                'approximation': approx,
                'error': error,
                'relative_error': error / abs(true_value) if true_value != 0 else float('inf')
            }
        
        # Composite methods
        composite_methods = {
            'Composite Trapezoidal': self.composite_trapezoidal,
            "Composite Simpson's 1/3": self.composite_simpsons_1_3,
            'Composite Midpoint': self.composite_midpoint
        }
        
        for name, method in composite_methods.items():
            results['composite_methods'][name] = []
            for n in n_values:
                # Skip odd n for Simpson's 1/3
                if name == "Composite Simpson's 1/3" and n % 2 != 0:
                    continue
                
                approx, _ = method(f, a, b, n)
                error = abs(approx - true_value)
                results['composite_methods'][name].append({
                    'n': n,
                    'approximation': approx,
                    'error': error,
                    'relative_error': error / abs(true_value) if true_value != 0 else float('inf')
                })
        
        return results
    
    def analyze_convergence(self, f: Callable, a: float, b: float,
                           true_value: float, n_max: int = 128) -> Dict:
        """
        Analyze convergence rate of composite methods.
        
        Parameters:
        -----------
        f : Callable
            Function to integrate
        a : float
            Lower limit
        b : float
            Upper limit
        true_value : float
            True value of the integral
        n_max : int
            Maximum number of subintervals
            
        Returns:
        --------
        Dict
            Convergence data for each method
        """
        n_values = [2**i for i in range(1, int(np.log2(n_max)) + 1)]
        
        convergence_data = {
            'n_values': n_values,
            'methods': {}
        }
        
        methods = {
            'Composite Trapezoidal': self.composite_trapezoidal,
            'Composite Midpoint': self.composite_midpoint
        }
        
        for name, method in methods.items():
            errors = []
            for n in n_values:
                approx, _ = method(f, a, b, n)
                error = abs(approx - true_value)
                errors.append(error)
            convergence_data['methods'][name] = errors
        
        # Simpson's 1/3 (only even n)
        simpson_errors = []
        simpson_n = [n for n in n_values if n % 2 == 0]
        for n in simpson_n:
            approx, _ = self.composite_simpsons_1_3(f, a, b, n)
            error = abs(approx - true_value)
            simpson_errors.append(error)
        
        convergence_data['methods']["Composite Simpson's 1/3"] = simpson_errors
        convergence_data['simpson_n'] = simpson_n
        
        return convergence_data


# ==================== UTILITY FUNCTIONS ====================

def print_basic_methods_table(results: Dict):
    """Print a formatted table of basic integration methods."""
    print("\n" + "="*80)
    print("BASIC INTEGRATION METHODS (Single Application)")
    print("="*80)
    print(f"{'Method':<25} {'Approximation':>15} {'Error':>15} {'Rel. Error':>15}")
    print("-"*80)
    
    for method, data in results['basic_methods'].items():
        print(f"{method:<25} {data['approximation']:>15.10f} "
              f"{data['error']:>15.2e} {data['relative_error']:>15.2e}")
    print("="*80)


def print_composite_methods_table(results: Dict):
    """Print a formatted table of composite integration methods."""
    print("\n" + "="*80)
    print("COMPOSITE INTEGRATION METHODS")
    print("="*80)
    
    for method, data_list in results['composite_methods'].items():
        print(f"\n{method}:")
        print(f"{'n':>6} {'Approximation':>18} {'Error':>15} {'Rel. Error':>15}")
        print("-"*80)
        for data in data_list:
            print(f"{data['n']:>6} {data['approximation']:>18.10f} "
                  f"{data['error']:>15.2e} {data['relative_error']:>15.2e}")
    print("="*80)


def plot_convergence(convergence_data: Dict, title: str = "Convergence Analysis"):
    """
    Plot error vs number of subintervals for composite methods.
    
    Parameters:
    -----------
    convergence_data : Dict
        Data from analyze_convergence method
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    n_values = convergence_data['n_values']
    
    for method, errors in convergence_data['methods'].items():
        if method == "Composite Simpson's 1/3":
            n_plot = convergence_data['simpson_n']
        else:
            n_plot = n_values
        
        plt.loglog(n_plot, errors, 'o-', label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Number of Subintervals (n)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_integration_visualization(f: Callable, a: float, b: float, n: int = 10,
                                   method: str = 'trapezoidal'):
    """
    Visualize how numerical integration approximates the area under a curve.
    
    Parameters:
    -----------
    f : Callable
        Function to integrate
    a : float
        Lower limit
    b : float
        Upper limit
    n : int
        Number of subintervals
    method : str
        'trapezoidal', 'simpson', or 'midpoint'
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the function
    x_smooth = np.linspace(a, b, 1000)
    y_smooth = [f(x) for x in x_smooth]
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='f(x)')
    
    # Create subintervals
    h = (b - a) / n
    x_points = np.linspace(a, b, n + 1)
    
    if method == 'trapezoidal':
        # Draw trapezoids
        for i in range(n):
            x_left = x_points[i]
            x_right = x_points[i + 1]
            y_left = f(x_left)
            y_right = f(x_right)
            
            # Draw trapezoid
            ax.fill([x_left, x_right, x_right, x_left],
                   [0, 0, y_right, y_left],
                   alpha=0.3, edgecolor='red', facecolor='yellow')
        
        title = f'Trapezoidal Rule Visualization (n={n})'
    
    elif method == 'midpoint':
        # Draw rectangles at midpoints
        for i in range(n):
            x_left = x_points[i]
            x_right = x_points[i + 1]
            x_mid = (x_left + x_right) / 2
            y_mid = f(x_mid)
            
            # Draw rectangle
            ax.fill([x_left, x_right, x_right, x_left],
                   [0, 0, y_mid, y_mid],
                   alpha=0.3, edgecolor='green', facecolor='lightgreen')
        
        title = f'Midpoint Rule Visualization (n={n})'
    
    elif method == 'simpson':
        # Draw parabolic segments (simplified visualization)
        for i in range(0, n, 2):
            if i + 2 <= n:
                x_seg = np.linspace(x_points[i], x_points[i + 2], 100)
                y_seg = [f(x) for x in x_seg]
                ax.fill_between(x_seg, 0, y_seg, alpha=0.3, color='orange')
        
        title = f"Simpson's 1/3 Rule Visualization (n={n})"
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ==================== DEMONSTRATION ====================

def demonstrate_lab3():
    """
    Comprehensive demonstration of Lab 3: Numerical Integration.
    """
    print("\n" + "="*80)
    print("LAB 3: NUMERICAL INTEGRATION - DEMONSTRATION")
    print("="*80)
    
    # Define test function: f(x) = x^2
    # True integral from 0 to 1: ∫x² dx = x³/3 |₀¹ = 1/3
    f = lambda x: x**2
    a, b = 0, 1
    true_value = 1/3
    
    print(f"\nTest Function: f(x) = x²")
    print(f"Interval: [{a}, {b}]")
    print(f"True Value: ∫f(x)dx = {true_value:.10f}")
    
    # Initialize
    ni = NumericalIntegration()
    
    # Compare all methods
    print("\n" + "-"*80)
    print("COMPARING ALL METHODS")
    print("-"*80)
    
    n_values = [2, 4, 8, 16, 32, 64]
    results = ni.compare_methods(f, a, b, true_value, n_values)
    
    # Print tables
    print_basic_methods_table(results)
    print_composite_methods_table(results)
    
    # Convergence analysis
    print("\n" + "-"*80)
    print("CONVERGENCE ANALYSIS")
    print("-"*80)
    
    convergence_data = ni.analyze_convergence(f, a, b, true_value, n_max=128)
    
    # Plot convergence
    plot_convergence(convergence_data, 
                    title="Error vs Number of Subintervals (f(x) = x²)")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_integration_visualization(f, a, b, n=8, method='trapezoidal')
    plot_integration_visualization(f, a, b, n=8, method='midpoint')
    plot_integration_visualization(f, a, b, n=8, method='simpson')
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demonstrate_lab3()
