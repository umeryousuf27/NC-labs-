"""
LAB 5: SOLVING LINEAR SYSTEMS (A·x = b)
========================================

This module implements methods for solving systems of linear equations:
- LU Decomposition (Doolittle method)
- Jacobi Iterative Method
- Gauss-Seidel Iterative Method

Author: Numerical Methods Lab
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass
import warnings


@dataclass
class IterativeResult:
    """Store results from iterative methods."""
    x_solution: np.ndarray
    iterations: int
    residuals: List[float]
    x_history: List[np.ndarray]
    converged: bool
    method_name: str


class LinearSystemSolvers:
    """
    A comprehensive class for solving linear systems A·x = b.
    
    Implements both direct (LU decomposition) and iterative (Jacobi, Gauss-Seidel) methods.
    """
    
    def __init__(self):
        """Initialize the linear system solvers class."""
        pass
    
    # ==================== DIAGONAL DOMINANCE CHECK ====================
    
    @staticmethod
    def check_diagonal_dominance(A: np.ndarray, strict: bool = True) -> Tuple[bool, str]:
        """
        Check if matrix A is diagonally dominant.
        
        A matrix is strictly diagonally dominant if:
        |a_ii| > Σ|a_ij| for all i (j ≠ i)
        
        A matrix is weakly diagonally dominant if:
        |a_ii| ≥ Σ|a_ij| for all i (j ≠ i)
        
        Diagonal dominance is a sufficient condition for convergence
        of Jacobi and Gauss-Seidel methods.
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        strict : bool
            If True, check strict diagonal dominance
            
        Returns:
        --------
        Tuple[bool, str]
            (is_dominant, message)
        """
        n = A.shape[0]
        
        for i in range(n):
            diagonal = abs(A[i, i])
            row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
            
            if strict:
                if diagonal <= row_sum:
                    return False, f"Row {i}: |{A[i,i]:.4f}| ≤ {row_sum:.4f} (not strictly dominant)"
            else:
                if diagonal < row_sum:
                    return False, f"Row {i}: |{A[i,i]:.4f}| < {row_sum:.4f} (not weakly dominant)"
        
        dominance_type = "strictly" if strict else "weakly"
        return True, f"Matrix is {dominance_type} diagonally dominant"
    
    # ==================== LU DECOMPOSITION ====================
    
    def lu_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LU Decomposition using Doolittle's method.
        
        Decomposes matrix A into lower triangular L and upper triangular U
        such that A = L·U
        
        Doolittle's method:
        - L has 1's on the diagonal
        - U has the actual diagonal values
        
        Algorithm:
        ----------
        For each row i:
            For each column j:
                If i ≤ j (upper triangle):
                    u_ij = a_ij - Σ(l_ik * u_kj) for k=0 to i-1
                If i > j (lower triangle):
                    l_ij = (a_ij - Σ(l_ik * u_kj)) / u_jj for k=0 to j-1
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix (must be square)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (L, U) matrices
            
        Raises:
        -------
        ValueError
            If matrix is singular or not square
        """
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        
        L = np.zeros((n, n))
        U = np.zeros((n, n))
        
        for i in range(n):
            # Upper triangular matrix U
            for j in range(i, n):
                sum_lu = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - sum_lu
            
            # Lower triangular matrix L
            L[i, i] = 1  # Doolittle: diagonal of L is 1
            for j in range(i + 1, n):
                if U[i, i] == 0:
                    raise ValueError(f"Matrix is singular (zero pivot at position {i})")
                sum_lu = sum(L[j, k] * U[k, i] for k in range(i))
                L[j, i] = (A[j, i] - sum_lu) / U[i, i]
        
        return L, U
    
    def forward_substitution(self, L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve L·y = b for y using forward substitution.
        
        L is lower triangular.
        
        Algorithm:
        ----------
        y_0 = b_0 / L_00
        For i = 1 to n-1:
            y_i = (b_i - Σ(L_ij * y_j)) / L_ii for j=0 to i-1
        
        Parameters:
        -----------
        L : np.ndarray
            Lower triangular matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        np.ndarray
            Solution vector y
        """
        n = len(b)
        y = np.zeros(n)
        
        for i in range(n):
            sum_ly = sum(L[i, j] * y[j] for j in range(i))
            y[i] = (b[i] - sum_ly) / L[i, i]
        
        return y
    
    def backward_substitution(self, U: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve U·x = y for x using backward substitution.
        
        U is upper triangular.
        
        Algorithm:
        ----------
        x_{n-1} = y_{n-1} / U_{n-1,n-1}
        For i = n-2 down to 0:
            x_i = (y_i - Σ(U_ij * x_j)) / U_ii for j=i+1 to n-1
        
        Parameters:
        -----------
        U : np.ndarray
            Upper triangular matrix
        y : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        np.ndarray
            Solution vector x
        """
        n = len(y)
        x = np.zeros(n)
        
        for i in range(n - 1, -1, -1):
            sum_ux = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_ux) / U[i, i]
        
        return x
    
    def solve_lu(self, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve A·x = b using LU decomposition.
        
        Process:
        --------
        1. Decompose A = L·U
        2. Solve L·y = b for y (forward substitution)
        3. Solve U·x = y for x (backward substitution)
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (x, L, U) - solution and decomposition matrices
        """
        L, U = self.lu_decomposition(A)
        y = self.forward_substitution(L, b)
        x = self.backward_substitution(U, y)
        
        return x, L, U
    
    # ==================== JACOBI ITERATIVE METHOD ====================
    
    def jacobi_method(self, A: np.ndarray, b: np.ndarray,
                     x0: np.ndarray = None, tol: float = 1e-6,
                     max_iter: int = 100) -> IterativeResult:
        """
        Jacobi Iterative Method for solving A·x = b.
        
        Formula:
        --------
        x_i^(k+1) = (b_i - Σ(a_ij * x_j^(k))) / a_ii  for j ≠ i
        
        Algorithm:
        ----------
        1. Start with initial guess x^(0)
        2. For each iteration k:
            For each component i:
                x_i^(k+1) = (b_i - Σ(a_ij * x_j^(k))) / a_ii
        3. Check convergence: ||x^(k+1) - x^(k)|| < tol
        4. Repeat until convergence or max iterations
        
        Characteristics:
        ----------------
        - Uses values from previous iteration only
        - All components updated simultaneously
        - Converges if matrix is strictly diagonally dominant
        - Can be parallelized
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x0 : np.ndarray, optional
            Initial guess (default: zeros)
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        IterativeResult
            Solution and convergence information
        """
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Check diagonal dominance
        is_dominant, msg = self.check_diagonal_dominance(A, strict=True)
        if not is_dominant:
            warnings.warn(f"Jacobi: {msg}. Convergence not guaranteed!")
        
        x_history = [x.copy()]
        residuals = []
        
        for iteration in range(max_iter):
            x_new = np.zeros(n)
            
            for i in range(n):
                sum_ax = sum(A[i, j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_ax) / A[i, i]
            
            # Calculate residual
            residual = np.linalg.norm(A @ x_new - b)
            residuals.append(residual)
            x_history.append(x_new.copy())
            
            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                return IterativeResult(
                    x_solution=x_new,
                    iterations=iteration + 1,
                    residuals=residuals,
                    x_history=x_history,
                    converged=True,
                    method_name="Jacobi Method"
                )
            
            x = x_new.copy()
        
        # Did not converge
        warnings.warn(f"Jacobi method did not converge in {max_iter} iterations")
        return IterativeResult(
            x_solution=x,
            iterations=max_iter,
            residuals=residuals,
            x_history=x_history,
            converged=False,
            method_name="Jacobi Method"
        )
    
    # ==================== GAUSS-SEIDEL ITERATIVE METHOD ====================
    
    def gauss_seidel_method(self, A: np.ndarray, b: np.ndarray,
                           x0: np.ndarray = None, tol: float = 1e-6,
                           max_iter: int = 100) -> IterativeResult:
        """
        Gauss-Seidel Iterative Method for solving A·x = b.
        
        Formula:
        --------
        x_i^(k+1) = (b_i - Σ(a_ij*x_j^(k+1)) - Σ(a_ij*x_j^(k))) / a_ii
                         j<i                j>i
        
        Algorithm:
        ----------
        1. Start with initial guess x^(0)
        2. For each iteration k:
            For each component i:
                Use already updated values x_j^(k+1) for j < i
                Use old values x_j^(k) for j > i
                x_i^(k+1) = (b_i - Σ(a_ij*x_j^(k+1)) - Σ(a_ij*x_j^(k))) / a_ii
        3. Check convergence
        4. Repeat until convergence
        
        Characteristics:
        ----------------
        - Uses most recent values immediately
        - Updates components sequentially
        - Generally converges faster than Jacobi
        - Cannot be parallelized easily
        - Converges if matrix is strictly diagonally dominant
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x0 : np.ndarray, optional
            Initial guess (default: zeros)
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        IterativeResult
            Solution and convergence information
        """
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Check diagonal dominance
        is_dominant, msg = self.check_diagonal_dominance(A, strict=True)
        if not is_dominant:
            warnings.warn(f"Gauss-Seidel: {msg}. Convergence not guaranteed!")
        
        x_history = [x.copy()]
        residuals = []
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            for i in range(n):
                # Use updated values for j < i, old values for j > i
                sum1 = sum(A[i, j] * x[j] for j in range(i))
                sum2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
                x[i] = (b[i] - sum1 - sum2) / A[i, i]
            
            # Calculate residual
            residual = np.linalg.norm(A @ x - b)
            residuals.append(residual)
            x_history.append(x.copy())
            
            # Check convergence
            if np.linalg.norm(x - x_old) < tol:
                return IterativeResult(
                    x_solution=x,
                    iterations=iteration + 1,
                    residuals=residuals,
                    x_history=x_history,
                    converged=True,
                    method_name="Gauss-Seidel Method"
                )
        
        # Did not converge
        warnings.warn(f"Gauss-Seidel method did not converge in {max_iter} iterations")
        return IterativeResult(
            x_solution=x,
            iterations=max_iter,
            residuals=residuals,
            x_history=x_history,
            converged=False,
            method_name="Gauss-Seidel Method"
        )
    
    # ==================== COMPARISON ====================
    
    def compare_iterative_methods(self, A: np.ndarray, b: np.ndarray,
                                  x0: np.ndarray = None, tol: float = 1e-6,
                                  max_iter: int = 100) -> Dict:
        """
        Compare Jacobi and Gauss-Seidel methods.
        
        Parameters:
        -----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side vector
        x0 : np.ndarray, optional
            Initial guess
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        Dict
            Results from both methods
        """
        jacobi_result = self.jacobi_method(A, b, x0, tol, max_iter)
        gauss_seidel_result = self.gauss_seidel_method(A, b, x0, tol, max_iter)
        
        return {
            'Jacobi': jacobi_result,
            'Gauss-Seidel': gauss_seidel_result
        }


# ==================== UTILITY FUNCTIONS ====================

def print_lu_decomposition(L: np.ndarray, U: np.ndarray, A: np.ndarray = None):
    """Print L and U matrices from LU decomposition."""
    print("\n" + "="*80)
    print("LU DECOMPOSITION")
    print("="*80)
    
    if A is not None:
        print("\nOriginal Matrix A:")
        print(A)
        print("\nVerification: L·U =")
        print(L @ U)
        print(f"\nMax difference: {np.max(np.abs(A - L @ U)):.2e}")
    
    print("\nLower Triangular Matrix L:")
    print(L)
    
    print("\nUpper Triangular Matrix U:")
    print(U)
    print("="*80)


def print_iteration_table(result: IterativeResult, show_every: int = 1):
    """Print iteration table for iterative methods."""
    print("\n" + "="*80)
    print(f"{result.method_name} - Iteration Table")
    print("="*80)
    print(f"Converged: {result.converged}")
    print(f"Total Iterations: {result.iterations}")
    print("="*80)
    
    n = len(result.x_solution)
    
    # Header
    header = f"{'Iter':>6} "
    for i in range(n):
        header += f"{'x' + str(i):>12} "
    header += f"{'Residual':>15}"
    print(header)
    print("-"*80)
    
    # Rows
    for k in range(0, len(result.x_history), show_every):
        if k < len(result.residuals):
            row = f"{k:>6} "
            for val in result.x_history[k]:
                row += f"{val:>12.6f} "
            row += f"{result.residuals[k]:>15.2e}"
            print(row)
    
    # Final solution
    if len(result.x_history) - 1 not in range(0, len(result.x_history), show_every):
        k = len(result.x_history) - 1
        row = f"{k:>6} "
        for val in result.x_history[k]:
            row += f"{val:>12.6f} "
        if k - 1 < len(result.residuals):
            row += f"{result.residuals[k-1]:>15.2e}"
        print(row)
    
    print("="*80)


def print_comparison_table(results: Dict):
    """Print comparison of iterative methods."""
    print("\n" + "="*80)
    print("COMPARISON: JACOBI vs GAUSS-SEIDEL")
    print("="*80)
    print(f"{'Method':<20} {'Iterations':>12} {'Converged':>12} {'Final Residual':>18}")
    print("-"*80)
    
    for name, result in results.items():
        final_res = result.residuals[-1] if result.residuals else float('nan')
        print(f"{name:<20} {result.iterations:>12} {str(result.converged):>12} {final_res:>18.2e}")
    
    print("="*80)
    
    # Print solutions
    print("\nFinal Solutions:")
    print("-"*80)
    n = len(results['Jacobi'].x_solution)
    print(f"{'Variable':<12} {'Jacobi':>15} {'Gauss-Seidel':>15} {'Difference':>15}")
    print("-"*80)
    for i in range(n):
        jac_val = results['Jacobi'].x_solution[i]
        gs_val = results['Gauss-Seidel'].x_solution[i]
        diff = abs(jac_val - gs_val)
        print(f"x{i:<11} {jac_val:>15.10f} {gs_val:>15.10f} {diff:>15.2e}")
    print("="*80)


def plot_convergence_comparison(results: Dict, title: str = "Convergence Comparison"):
    """Plot convergence of iterative methods."""
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        iterations = range(1, len(result.residuals) + 1)
        plt.semilogy(iterations, result.residuals, 'o-',
                    label=name, linewidth=2, markersize=6, alpha=0.7)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Residual ||Ax - b|| (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_solution_convergence(result: IterativeResult, true_solution: np.ndarray = None):
    """Plot how each component converges."""
    n = len(result.x_solution)
    
    fig, axes = plt.subplots(1, min(n, 3), figsize=(15, 5))
    if n == 1:
        axes = [axes]
    
    for i in range(min(n, 3)):
        x_i_history = [x[i] for x in result.x_history]
        iterations = range(len(x_i_history))
        
        axes[i].plot(iterations, x_i_history, 'o-', linewidth=2, markersize=6)
        
        if true_solution is not None:
            axes[i].axhline(y=true_solution[i], color='r', linestyle='--',
                          label=f'True: {true_solution[i]:.6f}')
            axes[i].legend()
        
        axes[i].set_xlabel('Iteration', fontsize=10)
        axes[i].set_ylabel(f'x{i}', fontsize=10)
        axes[i].set_title(f'Convergence of x{i}', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{result.method_name} - Component Convergence', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==================== DEMONSTRATION ====================

def demonstrate_lab5():
    """
    Comprehensive demonstration of Lab 5: Linear Systems.
    """
    print("\n" + "="*80)
    print("LAB 5: SOLVING LINEAR SYSTEMS (A·x = b) - DEMONSTRATION")
    print("="*80)
    
    # Test system (diagonally dominant for iterative methods)
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ], dtype=float)
    
    b = np.array([6, 25, -11, 15], dtype=float)
    
    print("\nTest System:")
    print("A =")
    print(A)
    print("\nb =", b)
    
    # Initialize solver
    solver = LinearSystemSolvers()
    
    # Check diagonal dominance
    print("\n" + "-"*80)
    print("DIAGONAL DOMINANCE CHECK")
    print("-"*80)
    is_dominant, msg = solver.check_diagonal_dominance(A)
    print(msg)
    
    # LU Decomposition
    print("\n" + "-"*80)
    print("METHOD 1: LU DECOMPOSITION")
    print("-"*80)
    
    x_lu, L, U = solver.solve_lu(A, b)
    print_lu_decomposition(L, U, A)
    
    print("\nSolution from LU Decomposition:")
    for i, val in enumerate(x_lu):
        print(f"x{i} = {val:.10f}")
    
    # Verify solution
    residual_lu = np.linalg.norm(A @ x_lu - b)
    print(f"\nResidual ||Ax - b|| = {residual_lu:.2e}")
    
    # Iterative Methods
    print("\n" + "-"*80)
    print("METHOD 2 & 3: ITERATIVE METHODS")
    print("-"*80)
    
    results = solver.compare_iterative_methods(A, b, tol=1e-8, max_iter=50)
    
    # Print iteration tables
    print_iteration_table(results['Jacobi'], show_every=5)
    print_iteration_table(results['Gauss-Seidel'], show_every=5)
    
    # Print comparison
    print_comparison_table(results)
    
    # Verify against LU solution
    print("\nComparison with LU Decomposition:")
    print("-"*80)
    for i in range(len(x_lu)):
        jac_diff = abs(results['Jacobi'].x_solution[i] - x_lu[i])
        gs_diff = abs(results['Gauss-Seidel'].x_solution[i] - x_lu[i])
        print(f"x{i}: Jacobi diff = {jac_diff:.2e}, Gauss-Seidel diff = {gs_diff:.2e}")
    
    # Plot convergence
    print("\nGenerating plots...")
    plot_convergence_comparison(results, "Jacobi vs Gauss-Seidel Convergence")
    plot_solution_convergence(results['Jacobi'], x_lu)
    plot_solution_convergence(results['Gauss-Seidel'], x_lu)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demonstrate_lab5()
