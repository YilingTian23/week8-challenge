import numpy as np
import scipy.sparse as sp

def custom_cg(A, b, tol, maxiter=5000):
    """Conjugate Gradient method with relative tolerance"""
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rsold = np.dot(r, r)
    norm_b = np.linalg.norm(b)
    
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        # Check relative tolerance
        if np.sqrt(rsnew) / norm_b < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, i + 1

def run_experiment():
    N_values = [32, 64, 128]
    tolerances = [1e-4, 1e-6, 1e-8]
    
    print("Testing CG tolerances across meshes (1D Poisson equation)")
    print("=" * 75)
    print(f"{'N':<5} | {'E_disc':<10} | {'Budget':<10} | {'E_alg (1e-4)':<12} | {'E_alg (1e-6)':<12} | {'E_alg (1e-8)':<12}")
    print("-" * 75)
    
    for N in N_values:
        h = 1.0 / (N + 1)
        x = np.linspace(h, 1 - h, N)
        
        # 1D Poisson: -u'' = f, with u = sin(pi*x)
        main_diag = 2.0 * np.ones(N) / h**2
        off_diag = -1.0 * np.ones(N - 1) / h**2
        A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
        
        b = np.pi**2 * np.sin(np.pi * x)
        u_exact = np.sin(np.pi * x)
        
        # Direct solve for reference
        u_ref = sp.linalg.spsolve(A, b)
        
        # Discretisation error (L2 norm)
        E_disc = np.linalg.norm(u_ref - u_exact) * np.sqrt(h)
        budget = 1e-3 * E_disc
        
        results = []
        iters = []
        for tol in tolerances:
            u_alg, iter_count = custom_cg(A, b, tol)
            E_alg = np.linalg.norm(u_alg - u_ref) * np.sqrt(h)
            results.append(E_alg)
            iters.append(iter_count)
            
        print(f"{N:<5} | {E_disc:.2e}   | {budget:.2e}   | {results[0]:.2e}      | {results[1]:.2e}      | {results[2]:.2e}")
        print(f"      Iterations: tol=1e-4 ({iters[0]}), tol=1e-6 ({iters[1]}), tol=1e-8 ({iters[2]})")
        print("-" * 75)

if __name__ == "__main__":
    run_experiment()