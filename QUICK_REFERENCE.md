# Quick Reference Guide
## Numerical Methods Lab 1 & 2

---

## 🔢 Lab 3: Numerical Integration

### Basic Rules
- **Trapezoidal:** $I \approx \frac{h}{2}[f(a) + f(b)]$
- **Simpson's 1/3:** $I \approx \frac{h}{3}[f(a) + 4f(\frac{a+b}{2}) + f(b)]$
- **Simpson's 3/8:** $I \approx \frac{3h}{8}[f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3)]$

### Composite Rules
- **Composite Trapezoidal:** $I \approx \frac{h}{2}[f(x_0) + 2\sum_{i=1}^{n-1}f(x_i) + f(x_n)]$
- **Composite Simpson's 1/3:** $I \approx \frac{h}{3}[f(x_0) + 4\sum_{odd}f(x_i) + 2\sum_{even}f(x_i) + f(x_n)]$

---

## 🔄 Lab 4: ODE Solvers

Problem: $\frac{dy}{dx} = f(x,y), \quad y(x_0) = y_0$

### Methods
1. **Euler (Order 1):**
   $y_{n+1} = y_n + h f(x_n, y_n)$

2. **Midpoint (Order 2):**
   $k_1 = f(x_n, y_n)$
   $k_2 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}k_1)$
   $y_{n+1} = y_n + h k_2$

3. **Modified Euler (Order 2):**
   $k_1 = f(x_n, y_n)$
   $k_2 = f(x_n + h, y_n + h k_1)$
   $y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)$

4. **Runge-Kutta 4 (Order 4):**
   $k_1 = f(x_n, y_n)$
   $k_2 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}k_1)$
   $k_3 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}k_2)$
   $k_4 = f(x_n + h, y_n + h k_3)$
   $y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

---

## 📐 Lab 5: Linear Systems

Problem: $A\mathbf{x} = \mathbf{b}$

### Direct Methods
- **LU Decomposition:** $A = LU$
  1. Solve $L\mathbf{y} = \mathbf{b}$ (Forward)
  2. Solve $U\mathbf{x} = \mathbf{y}$ (Backward)

### Iterative Methods
- **Jacobi:**
  $x_i^{(k+1)} = \frac{1}{a_{ii}}(b_i - \sum_{j \neq i} a_{ij}x_j^{(k)})$

- **Gauss-Seidel:**
  $x_i^{(k+1)} = \frac{1}{a_{ii}}(b_i - \sum_{j < i} a_{ij}x_j^{(k+1)} - \sum_{j > i} a_{ij}x_j^{(k)})$

---

## 🔍 Lab 1: Root Finding Methods

### When to Use Each Method

| Method | Use When | Avoid When |
|--------|----------|------------|
| **Bisection** | Need guaranteed convergence, have bracketing interval | Speed is critical |
| **False Position** | Want faster than bisection, have bracketing interval | Function is highly curved |
| **Fixed-Point** | Equation naturally rearranges to x = g(x) | \|g'(x)\| ≥ 1 |
| **Newton-Raphson** | Derivative available, need fast convergence | Derivative hard to compute |
| **Secant** | Want speed without derivative | Poor initial guesses |

### Quick Code Examples

```python
from lab1_root_finding import RootFindingMethods

methods = RootFindingMethods()
f = lambda x: x**3 - 2*x - 5

# Bisection
result = methods.bisection(f, a=2, b=3, tol=1e-6)

# Newton-Raphson
df = lambda x: 3*x**2 - 2
result = methods.newton_raphson(f, df, x0=2.0, tol=1e-6)

# Secant
result = methods.secant(f, x0=2.0, x1=3.0, tol=1e-6)

print(f"Root: {result['root']:.8f}")
print(f"Iterations: {result['iterations']}")
```

### Convergence Rates

- **Linear:** Error reduces by constant factor each iteration
  - Bisection, Fixed-Point
  
- **Superlinear:** Faster than linear, slower than quadratic
  - False Position, Secant (order ≈ 1.618)
  
- **Quadratic:** Error squares each iteration (very fast!)
  - Newton-Raphson

---

## 📊 Lab 2: Interpolation Methods

### Method Selection Guide

| Method | Best For | Data Spacing |
|--------|----------|--------------|
| **Lagrange** | Small datasets, any degree | Any spacing |
| **Newton DD** | Extending polynomial, general use | Any spacing |
| **Newton Forward** | Interpolation near start of table | Equal spacing |
| **Newton Backward** | Interpolation near end of table | Equal spacing |

### Quick Code Examples

```python
from lab2_interpolation import InterpolationMethods
import numpy as np

interp = InterpolationMethods()

# Lagrange Interpolation
x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 4.0, 9.0])
poly = interp.lagrange_interpolation(x, y)
print(f"P(2.5) = {poly(2.5):.4f}")

# Newton Divided Difference
poly, table = interp.newton_divided_difference(x, y)
interp.print_divided_difference_table(x, y)

# Newton Forward (equally spaced)
x_eq = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_eq = np.exp(x_eq)
poly_fwd, _ = interp.newton_forward(x_eq, y_eq)
print(f"P(0.25) = {poly_fwd(0.25):.4f}")  # Near start

# Newton Backward (equally spaced)
poly_bwd, _ = interp.newton_backward(x_eq, y_eq)
print(f"P(1.75) = {poly_bwd(1.75):.4f}")  # Near end
```

---

## 📐 Mathematical Formulas

### Root Finding

**Bisection:**
```
c = (a + b) / 2
if f(a)·f(c) < 0: b = c
else: a = c
```

**False Position:**
```
c = b - f(b)·(b - a) / (f(b) - f(a))
```

**Fixed-Point:**
```
x_{n+1} = g(x_n)
Converges if |g'(x)| < 1
```

**Newton-Raphson:**
```
x_{n+1} = x_n - f(x_n) / f'(x_n)
```

**Secant:**
```
x_{n+1} = x_n - f(x_n)·(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
```

### Interpolation

**Lagrange:**
```
P(x) = Σ y_i · L_i(x)
L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
```

**Newton Divided Difference:**
```
P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...
f[x_i,x_{i+1}] = (f[x_{i+1}] - f[x_i]) / (x_{i+1} - x_i)
```

**Newton Forward:**
```
P(x) = y_0 + u·Δy_0 + u(u-1)/2!·Δ²y_0 + ...
u = (x - x_0) / h
Δy_i = y_{i+1} - y_i
```

**Newton Backward:**
```
P(x) = y_n + v·∇y_n + v(v+1)/2!·∇²y_n + ...
v = (x - x_n) / h
∇y_i = y_i - y_{i-1}
```

---

## 🎯 Common Pitfalls

### Root Finding

❌ **Don't:**
- Use bisection without checking sign change
- Use Newton-Raphson when f'(x) = 0
- Choose poor initial guesses for open methods
- Ignore convergence status

✅ **Do:**
- Verify bracketing interval for bisection/false position
- Check derivative values for Newton-Raphson
- Use multiple initial guesses to test
- Always check if method converged

### Interpolation

❌ **Don't:**
- Use high-degree polynomials (Runge's phenomenon)
- Extrapolate far beyond data range
- Use forward difference near end of table
- Use backward difference near start of table

✅ **Do:**
- Keep polynomial degree reasonable (< 10)
- Interpolate within data range
- Use forward difference near start
- Use backward difference near end
- Verify polynomial passes through all points

---

## 📊 Typical Results

### Root Finding Example
**Problem:** x³ - 2x - 5 = 0 on [2, 3]

| Method | Iterations | Root |
|--------|-----------|------|
| Bisection | 20 | 2.094552 |
| Newton-Raphson | 4 | 2.094551 |
| Secant | 6 | 2.094551 |

### Interpolation Example
**Data:** (0,1), (1,2), (2,5), (3,10)

All methods give same polynomial:
- P(1.5) ≈ 3.25
- Degree: 3 (cubic)

---

## 🔧 Troubleshooting

### "Method not converging"
- Check initial guess/interval
- Verify function is continuous
- Increase max_iter
- Try different method

### "Division by zero"
- Newton: f'(x) = 0, choose different x0
- Secant: f(x0) = f(x1), choose different points

### "Overflow error"
- Fixed-point: |g'(x)| ≥ 1, rearrange equation differently
- Check if x values are reasonable

### "Interpolation looks wrong"
- Verify data points are correct
- Check if using right method for data spacing
- Plot to visualize
- Reduce polynomial degree

---

## 📈 Performance Tips

### Speed Optimization
1. Use Newton-Raphson when derivative available
2. Use Secant when derivative not available
3. Use lower tolerance if high precision not needed
4. Cache function evaluations if expensive

### Accuracy Improvement
1. Use tighter tolerance (e.g., 1e-10)
2. Use double precision (default in Python)
3. For interpolation, use more data points
4. Avoid high-degree polynomials

---

## 🎓 Study Tips

### For Exams
1. **Know convergence rates:**
   - Bisection: Linear
   - Newton: Quadratic
   - Secant: Superlinear (1.618)

2. **Know when each method fails:**
   - Bisection: No sign change
   - Newton: f'(x) = 0
   - Fixed-point: |g'(x)| ≥ 1

3. **Know interpolation uniqueness:**
   - Polynomial of degree ≤ n-1 through n points is unique
   - All methods give same polynomial

4. **Practice hand calculations:**
   - First 2-3 iterations of each method
   - Build divided difference table
   - Evaluate Lagrange basis polynomials

### For Implementation
1. Always include error handling
2. Return iteration history for analysis
3. Provide visualization functions
4. Test with known examples
5. Document assumptions

---

## 📚 Additional Resources

### Books
- Burden & Faires: "Numerical Analysis"
- Chapra & Canale: "Numerical Methods for Engineers"

### Online
- NumPy documentation: https://numpy.org/doc/
- Matplotlib gallery: https://matplotlib.org/gallery/
- SciPy optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html

### Practice Problems
1. Find roots of: sin(x) = x/2
2. Find roots of: e^x = 3x
3. Interpolate: f(x) = ln(x) with 5 points
4. Compare forward vs backward difference accuracy

---

## ✅ Quick Checklist

Before submitting/using code:

- [ ] All methods implemented
- [ ] Docstrings complete
- [ ] Test cases pass
- [ ] Plots generate correctly
- [ ] Tables format properly
- [ ] Error handling included
- [ ] Code follows style guide
- [ ] Results verified
- [ ] Documentation clear

---

**Quick Reference Version 1.0**  
**Date:** November 2025  
**For:** Numerical Methods Lab 1 & 2
