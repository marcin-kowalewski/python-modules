"""
Moduł zawierający metody numeryczne do rozwiązywania równań nieliniowych:
bisekcja, newtona i siecznych.
"""
import math
		
def bisection(f, a, b, tol=1e-12, max_iter=200):
		fa = f(a)
		fb = f(b)
		
	if fa == 0:
		return {
			"root": a,
			"iterations": 0,
			"residual": 0.0,
			"converged": True,
			"history": [a]
		}
		
	if fb == 0:
		return {
			"root": b,
			"iterations": 0,
			"residual": 0.0,
			"converged": True,
			"history": [b]
		}
		
	if fa * fb > 0:
		raise ValueError("Brak zmiany znaku na koncach przedzialu.")
		
		history = []
		
	for k in range(1, max_iter + 1):
		m = (a + b) / 2.0
		fm = f(m)
		
		history.append(m)
		
		if abs(fm) < tol or (b - a) / 2.0 < tol:
			return {
				"root": m,
				"iterations": k,
				"residual": abs(fm),
				"converged": True,
				"history": history
			}
			
		if fa * fm < 0:
			b = m
			fb = fm
		else:
			a = m
			fa = fm
		
		return {
			"root": m,
			"iterations": max_iter,
			"residual": abs(fm),
			"converged": False,
			"history": history
		}
		
		
def newton(f, df, x0, tol=1e-12, max_iter=100):
		x = float(x0)
		history = [x]
		
	for k in range(1, max_iter + 1):
		fx = f(x)
		
		if abs(fx) < tol:
			return {
				"root": x,
				"iterations": k - 1,
				"residual": abs(fx),
				"converged": True,
				"history": history
			}
			
			dfx = df(x)
			
		if abs(dfx) < 1e-14:
			return {
				"root": x,
				"iterations": k - 1,
				"residual": abs(fx),
				"converged": False,
				"history": history
			}
			
			x_new = x - fx / dfx
			history.append(x_new)
			
		if abs(x_new - x) < tol:
			x = x_new
			return {
				"root": x,
				"iterations": k,
				"residual": abs(f(x)),
				"converged": True,
				"history": history
			}
			
			x = x_new
		
		return {
			"root": x,
			"iterations": max_iter,
			"residual": abs(f(x)),
			"converged": False,
			"history": history
		}
		
		
def secant(f, x0, x1, tol=1e-12, max_iter=100):
		x0 = float(x0)
		x1 = float(x1)
		
		history = [x0, x1]
		
	for k in range(1, max_iter + 1):
		f0 = f(x0)
		f1 = f(x1)
		
		if abs(f1) < tol:
			return {
				"root": x1,
				"iterations": k - 1,
				"residual": abs(f1),
				"converged": True,
				"history": history
			}
			
			denominator = f1 - f0
			
		if abs(denominator) < 1e-14:
			return {
				"root": x1,
				"iterations": k - 1,
				"residual": abs(f1),
				"converged": False,
				"history": history
			}
			
			x2 = x1 - f1 * (x1 - x0) / denominator
			history.append(x2)
			
		if abs(x2 - x1) < tol:
			return {
				"root": x2,
				"iterations": k,
				"residual": abs(f(x2)),
				"converged": True,
				"history": history
			}
			
			x0, x1 = x1, x2
		
		return {
			"root": x1,
			"iterations": max_iter,
			"residual": abs(f(x1)),
			"converged": False,
			"history": history
		}
