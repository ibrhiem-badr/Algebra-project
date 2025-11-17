# save as gauss_gui_clean.py and run: python gauss_gui_clean.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np

# ------------------ Helpers ------------------
def fmt_num(x):
    """يعرض الرقم بدون كسور إلا لو فيها كسر فعلاً"""
    if abs(x - int(x)) < 1e-10:
        return str(int(round(x)))
    else:
        return f"{x:.3f}".rstrip('0').rstrip('.')

def fmt_matrix_text(M):
    rows, cols = M.shape
    lines = []
    for r in range(rows):
        left = "  ".join(f"{fmt_num(M[r,c]):>6}" for c in range(cols-1))
        right = f"{fmt_num(M[r,-1]):>6}"
        lines.append(f"[ {left} | {right} ]")
    return "\n".join(lines)

class Logger:
    def __init__(self, text_widget):
        self.text = text_widget
    def write(self, msg):
        self.text.insert(tk.END, msg)
        self.text.see(tk.END)
    def flush(self): pass

# ------------------ Gaussian ------------------
def gaussian_elimination_steps(A, b, log_func):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])
    log_func("\nInitial augmented matrix:\n")
    log_func(fmt_matrix_text(M) + "\n\n")

    # Forward elimination
    for i in range(n):
        piv_row = max(range(i, n), key=lambda r: abs(M[r,i]))
        if abs(M[piv_row,i]) < 1e-12:
            raise ValueError("Pivot nearly zero → singular system.")
        if piv_row != i:
            M[[i,piv_row]] = M[[piv_row,i]]
            log_func(f"Swap R{i+1} ↔ R{piv_row+1}\n")
            log_func(fmt_matrix_text(M) + "\n\n")

        for j in range(i+1, n):
            factor = M[j,i] / M[i,i]
            if abs(factor) > 1e-12:
                M[j,i:] -= factor * M[i,i:]
                f = fmt_num(factor)
                log_func(f"R{j+1} - {f}R{i+1} → R{j+1}\n")
                log_func(fmt_matrix_text(M) + "\n\n")

    # Back substitution
    x = np.zeros(n)
    log_func("--- Back substitution ---\n")
    for i in range(n-1, -1, -1):
        rhs = M[i,-1] - np.dot(M[i,i+1:], x[i+1:])
        x[i] = rhs / M[i,i]
        log_func(f"x{i+1} = {fmt_num(x[i])}\n")
    log_func("\nSolution (Gaussian):\n")
    for i,v in enumerate(x,1):
        log_func(f"x{i} = {fmt_num(v)}\n")
    return x

# ------------------ Gauss-Jordan ------------------
def gauss_jordan_steps(A, b, log_func):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])
    log_func("\nInitial augmented matrix:\n")
    log_func(fmt_matrix_text(M) + "\n\n")

    for i in range(n):
        piv_row = max(range(i, n), key=lambda r: abs(M[r,i]))
        if abs(M[piv_row,i]) < 1e-12:
            raise ValueError("Pivot nearly zero → singular system.")
        if piv_row != i:
            M[[i,piv_row]] = M[[piv_row,i]]
            log_func(f"Swap R{i+1} ↔ R{piv_row+1}\n")
            log_func(fmt_matrix_text(M) + "\n\n")

        pivot = M[i,i]
        if abs(pivot - 1) > 1e-12:
            M[i,:] /= pivot
            log_func(f"R{i+1} ÷ {fmt_num(pivot)} → R{i+1}\n")
            log_func(fmt_matrix_text(M) + "\n\n")

        for j in range(n):
            if j != i:
                factor = M[j,i]
                if abs(factor) > 1e-12:
                    M[j,:] -= factor * M[i,:]
                    log_func(f"R{j+1} - {fmt_num(factor)}R{i+1} → R{j+1}\n")
                    log_func(fmt_matrix_text(M) + "\n\n")

    x = M[:,-1]
    log_func("Solution (Gauss-Jordan):\n")
    for i,v in enumerate(x,1):
        log_func(f"x{i} = {fmt_num(v)}\n")
    return x

# ------------------ GUI ------------------
class GaussApp:
    def __init__(self, root):
        self.root = root
        root.title("Gaussian & Gauss-Jordan Step Solver")
        root.geometry("900x700")

        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="عدد المعادلات / المتغيرات (n):").pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=3)
        self.n_entry = ttk.Entry(ctrl, textvariable=self.n_var, width=4)
        self.n_entry.pack(side=tk.LEFT, padx=(6,12))

        ttk.Button(ctrl, text="Create Grid", command=self.create_grid).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Run Gaussian", command=self.run_gaussian).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Run Gauss-Jordan", command=self.run_gj).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Run Both", command=self.run_both).pack(side=tk.LEFT, padx=6)

        self.grid_frame = ttk.Frame(root, padding=8)
        self.grid_frame.pack(fill=tk.X)

        self.output = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=28, font=("Consolas", 10))
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.entries = []
        self.create_grid()

    def create_grid(self):
        n = self.n_var.get()
        if n <= 0 or n > 15:
            messagebox.showerror("Error", "ادخل n من 1 لحد 15")
            return
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.entries = []
        ttk.Label(self.grid_frame, text="أدخل معاملات كل معادلة (المعاملات ثم الثابت):").pack(anchor=tk.W)
        for i in range(n):
            rowframe = ttk.Frame(self.grid_frame)
            rowframe.pack(fill=tk.X, pady=2)
            ttk.Label(rowframe, text=f"المعادلة {i+1}:").pack(side=tk.LEFT)
            e = ttk.Entry(rowframe, width=80)
            e.pack(side=tk.LEFT, padx=6)
            e.insert(0, " ".join("0" for _ in range(n+1)))
            self.entries.append(e)

    def clear_results(self):
        self.output.delete("1.0", tk.END)

    def read_matrix(self):
        try:
            n = self.n_var.get()
            A = np.zeros((n,n))
            b = np.zeros(n)
            for i, e in enumerate(self.entries):
                parts = e.get().strip().split()
                if len(parts) != n+1:
                    raise ValueError(f"Row {i+1} must have {n+1} numbers.")
                nums = [float(x) for x in parts]
                A[i,:] = nums[:n]
                b[i] = nums[-1]
            return A, b
        except Exception as ex:
            messagebox.showerror("Input error", str(ex))
            return None, None

    def run_gaussian(self):
        A,b = self.read_matrix()
        if A is None: return
        self.output.insert(tk.END, "\n=== Running Gaussian Elimination ===\n")
        logger = Logger(self.output)
        try:
            gaussian_elimination_steps(A.copy(), b.copy(), logger.write)
        except Exception as ex:
            self.output.insert(tk.END, f"\nError: {ex}\n")

    def run_gj(self):
        A,b = self.read_matrix()
        if A is None: return
        self.output.insert(tk.END, "\n=== Running Gauss-Jordan Elimination ===\n")
        logger = Logger(self.output)
        try:
            gauss_jordan_steps(A.copy(), b.copy(), logger.write)
        except Exception as ex:
            self.output.insert(tk.END, f"\nError: {ex}\n")

    def run_both(self):
        self.run_gaussian()
        self.run_gj()

# ------------------ Run ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GaussApp(root)
    root.mainloop()
