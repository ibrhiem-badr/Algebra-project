import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np


def fmt_num(x):
    if abs(x - int(x)) < 1e-9:
        return str(int(round(x)))
    else:
        return f"{x:.2f}".rstrip('0').rstrip('.')


def fmt_matrix_text(M):
    rows, cols = M.shape
    lines = []
    for r in range(rows):
        left_parts = [f"{fmt_num(M[r, c]):>6}" for c in range(cols - 1)]
        left = "  ".join(left_parts)
        right = f"{fmt_num(M[r, -1]):>6}"
        lines.append(f"[ {left} | {right} ]")
    return "\n".join(lines)


class Logger:
    def __init__(self, text_widget):
        self.text = text_widget

    def write(self, msg, tag=None):
        self.text.insert(tk.END, msg, tag)
        self.text.see(tk.END)


def extract_parametric_equations(M, n_vars, log_func):
    rows, _ = M.shape
    pivots = {}

    for r in range(rows):
        for c in range(n_vars):
            if abs(M[r, c]) > 1e-9:
                pivots[c] = r
                break

    final_solutions = {}

    log_func("\n--- Deriving Equations ---\n", "header")

    for j in range(n_vars):
        var_name = f"x{j + 1}"

        if j not in pivots:
            final_solutions[var_name] = f"{var_name} (Free)"
            continue

        r = pivots[j]
        sol_parts = []

        for c in range(j + 1, n_vars):
            coeff = M[r, c]
            if abs(coeff) > 1e-9:
                new_coeff = -coeff
                sign = "+" if new_coeff > 0 else "-"
                val = abs(new_coeff)
                val_str = fmt_num(val)
                if val_str == "1": val_str = ""

                if not sol_parts and sign == "+":
                    sol_parts.append(f"{val_str}x{c + 1}")
                else:
                    sol_parts.append(f"{sign} {val_str}x{c + 1}")

        val_b = M[r, -1]
        if abs(val_b) > 1e-9:
            sign = "+" if val_b > 0 else "-"
            val_str = fmt_num(abs(val_b))

            if not sol_parts and sign == "+":
                sol_parts.append(val_str)
            elif not sol_parts and sign == "-":
                sol_parts.append(f"-{val_str}")
            else:
                sol_parts.append(f"{sign} {val_str}")

        if not sol_parts:
            rhs = "0"
        else:
            rhs = " ".join(sol_parts)

        final_solutions[var_name] = rhs
        log_func(f"From R{r + 1}:  {var_name} = {rhs}\n")

    log_func("\n" + "=" * 40 + "\n")
    log_func("Final Solution Set (Ordered):\n", "result")
    log_func("=" * 40 + "\n")

    for j in range(n_vars):
        v = f"x{j + 1}"
        sol = final_solutions[v]
        if "Free" in sol:
            log_func(f"{v} = {sol}\n", "info")
        else:
            log_func(f"{v} = {sol}\n", "result")

    log_func("=" * 40 + "\n\n")


def check_solution_status(M, n_vars, log_func):
    rows, total_cols = M.shape
    has_no_solution = False
    pivot_count = 0

    for i in range(rows):
        row_A = M[i, :n_vars]
        val_b = M[i, -1]
        is_all_zeros = np.all(np.abs(row_A) < 1e-9)

        if is_all_zeros:
            if abs(val_b) > 1e-9:
                has_no_solution = True
        else:
            pivot_count += 1

    if has_no_solution:
        log_func("\n-----------------------------------\n")
        log_func("The System has no solution\n", "error")
        log_func("-----------------------------------\n")
        return "none"
    elif pivot_count < n_vars:
        free_vars = n_vars - pivot_count
        log_func("\n-----------------------------------\n")
        log_func("The System has infinitely many solutions\n", "info")
        log_func(f"(Rank = {pivot_count}, Free Variables = {free_vars})\n")
        extract_parametric_equations(M, n_vars, log_func)
        log_func("-----------------------------------\n")
        return "infinite"
    return "unique"


def textbook_solve(A, b, log_func, method="gaussian"):
    rows, cols = A.shape
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    log_func(f"=== Solve by {method.title()} ({rows}x{cols}) ===\n\n", "header")
    log_func("Initial Augmented Matrix:\n")
    log_func(fmt_matrix_text(M) + "\n\n")

    limit = min(rows, cols)

    for i in range(limit):
        pivot = M[i, i]
        if abs(pivot) < 1e-9:
            swapped = False
            for k in range(i + 1, rows):
                if abs(M[k, i]) > 1e-9:
                    M[[i, k]] = M[[k, i]]
                    log_func(f"Interchange R{i + 1} with R{k + 1}\n", "action")
                    log_func(fmt_matrix_text(M) + "\n\n")
                    pivot = M[i, i]
                    swapped = True
                    break
            if not swapped and abs(pivot) < 1e-9:
                continue

        if abs(pivot - 1) > 1e-9:
            divisor = pivot
            M[i, :] = M[i, :] / divisor
            d_str = fmt_num(divisor)
            log_func(f"R{i + 1} / {d_str} → R{i + 1}\n", "action")
            log_func(fmt_matrix_text(M) + "\n\n")

        rows_to_check = range(rows) if method == "gauss-jordan" else range(i + 1, rows)

        for k in rows_to_check:
            if k == i: continue
            val_to_kill = M[k, i]
            if abs(val_to_kill) > 1e-9:
                multiplier = -val_to_kill
                M[k, :] = M[k, :] + multiplier * M[i, :]

                m_str = fmt_num(multiplier)
                if abs(multiplier - 1) < 1e-9:
                    m_str = ""
                elif abs(multiplier + 1) < 1e-9:
                    m_str = "-"
                elif multiplier > 0:
                    m_str = f"+{m_str}"

                if multiplier < 0:
                    log_func(f"R{k + 1} {m_str}R{i + 1} → R{k + 1}\n", "action")
                else:
                    op = "+" if multiplier > 0 else ""
                    clean_mult = fmt_num(abs(multiplier))
                    if clean_mult == "1": clean_mult = ""
                    log_func(f"R{k + 1} {op} {clean_mult}R{i + 1} → R{k + 1}\n", "action")
                log_func(fmt_matrix_text(M) + "\n\n")

    status = check_solution_status(M, cols, log_func)
    if status != "unique": return

    if method == "gaussian" and rows == cols:
        x = np.zeros(cols)
        log_func("--- Back Substitution ---\n", "header")
        for i in range(rows - 1, -1, -1):
            rhs = M[i, -1] - np.dot(M[i, i + 1:cols], x[i + 1:])
            x[i] = rhs
            log_func(f"x{i + 1} = {fmt_num(x[i])}\n")
    elif method == "gauss-jordan" and rows == cols:
        x = M[:, -1]

    if rows == cols:
        log_func("\nSolution Set:\n", "header")
        for i, v in enumerate(x, 1):
            log_func(f"x{i} = {fmt_num(v)}\n", "result")


class GaussApp:
    def __init__(self, root):
        self.root = root
        root.title("Matrix Solver Pro")

        w, h = 1100, 850
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        bg_color = "#f4f6f9"
        root.configure(bg=bg_color)

        self.style = ttk.Style()
        self.style.theme_use('clam')

        primary_color = "#2980b9"
        accent_color = "#27ae60"
        danger_color = "#c0392b"
        dark_text = "#2c3e50"

        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabelframe", background=bg_color, relief="flat")
        self.style.configure("TLabelframe.Label", background=bg_color, foreground=primary_color,
                             font=("Segoe UI", 11, "bold"))
        self.style.configure("TLabel", background=bg_color, foreground=dark_text, font=("Segoe UI", 10))

        self.style.configure("TButton", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6, background="#bdc3c7",
                             foreground="#2c3e50")
        self.style.map("TButton", background=[('active', '#95a5a6')])
        self.style.configure("Primary.TButton", background=primary_color, foreground="white")
        self.style.map("Primary.TButton", background=[('active', '#3498db')])
        self.style.configure("Accent.TButton", background=accent_color, foreground="white")
        self.style.map("Accent.TButton", background=[('active', '#2ecc71')])
        self.style.configure("Danger.TButton", background=danger_color, foreground="white")
        self.style.map("Danger.TButton", background=[('active', '#e74c3c')])

        header_frame = tk.Frame(root, bg=primary_color, pady=15)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="Linear Algebra System Solver", font=("Segoe UI", 16, "bold"), bg=primary_color,
                 fg="white").pack()

        ctrl = ttk.Frame(root, padding=(20, 15, 20, 5))
        ctrl.pack(fill=tk.X)

        dim_frame = ttk.LabelFrame(ctrl, text=" Dimensions ", padding=10)
        dim_frame.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(dim_frame, text="Equations (Rows):").pack(side=tk.LEFT)
        self.rows_var = tk.IntVar(value=3)
        rows_ent = ttk.Entry(dim_frame, textvariable=self.rows_var, width=4, font=("Segoe UI", 10))
        rows_ent.pack(side=tk.LEFT, padx=5)

        ttk.Label(dim_frame, text="Variables (Cols):").pack(side=tk.LEFT, padx=(10, 0))
        self.cols_var = tk.IntVar(value=4)
        cols_ent = ttk.Entry(dim_frame, textvariable=self.cols_var, width=4, font=("Segoe UI", 10))
        cols_ent.pack(side=tk.LEFT, padx=5)

        rows_ent.bind("<Return>", lambda e: self.create_grid())
        cols_ent.bind("<Return>", lambda e: self.create_grid())

        act_frame = ttk.LabelFrame(ctrl, text=" Actions ", padding=10)
        act_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(act_frame, text="Create Grid", command=self.create_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(act_frame, text="Clear Log", command=self.clear_results, style="Danger.TButton").pack(side=tk.LEFT,
                                                                                                         padx=5)

        solve_frame = ttk.LabelFrame(ctrl, text=" Solve Method ", padding=10)
        solve_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(solve_frame, text="Gaussian", command=lambda: self.run_solver("gaussian"),
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(solve_frame, text="Gauss-Jordan", command=lambda: self.run_solver("gauss-jordan"),
                   style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(solve_frame, text="Run Both", command=self.run_both, style="Accent.TButton").pack(side=tk.LEFT,
                                                                                                     padx=5)

        self.input_frame_container = ttk.LabelFrame(root, text=" Matrix Input (A | b) ", padding=10)
        self.input_frame_container.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)

        self.canvas = tk.Canvas(self.input_frame_container, bg=bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.input_frame_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(height=150)

        self.matrix_entries = []
        self.linear_widgets = []
        self.create_grid()

        self.out_frame = ttk.LabelFrame(root, text=" Solution Steps ", padding=10)
        self.out_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        self.output = scrolledtext.ScrolledText(self.out_frame, height=30, font=("Consolas", 11),
                                                state='normal', bd=1, relief="solid")
        self.output.pack(fill=tk.BOTH, expand=True)

        self.output.tag_config("header", foreground="#2980b9", font=("Consolas", 11, "bold"))
        self.output.tag_config("action", foreground="#27ae60")
        self.output.tag_config("error", foreground="#c0392b", font=("Consolas", 12, "bold"))
        self.output.tag_config("info", foreground="#8e44ad", font=("Consolas", 12, "bold"))
        self.output.tag_config("result", foreground="#2980b9", font=("Consolas", 11, "bold"), background="#ecf0f1")
        self.output.tag_config("sep", foreground="#d35400", font=("Consolas", 12, "bold"))

    def create_grid(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.matrix_entries = []
        self.linear_widgets = []

        try:
            rows = self.rows_var.get()
            cols = self.cols_var.get()
        except:
            return

        if rows <= 0 or cols <= 0 or rows > 20 or cols > 20:
            messagebox.showerror("Error", "Please keep dimensions between 1 and 20")
            return

        for j in range(cols):
            lbl = ttk.Label(self.scrollable_frame, text=f"x{j + 1}", font=("Segoe UI", 10, "bold"))
            lbl.grid(row=0, column=j, padx=2, pady=5)

        ttk.Label(self.scrollable_frame, text="=").grid(row=0, column=cols, padx=5)
        ttk.Label(self.scrollable_frame, text="b", font=("Segoe UI", 10, "bold")).grid(row=0, column=cols + 1, padx=2)

        for i in range(rows):
            row_ents = []
            for j in range(cols):
                e = ttk.Entry(self.scrollable_frame, width=8, justify="center", font=("Consolas", 11))
                e.grid(row=i + 1, column=j, padx=4, pady=4)
                row_ents.append(e)
                self.linear_widgets.append(e)

            ttk.Label(self.scrollable_frame, text="|").grid(row=i + 1, column=cols)

            b_ent = ttk.Entry(self.scrollable_frame, width=8, justify="center", font=("Consolas", 11))
            b_ent.grid(row=i + 1, column=cols + 1, padx=4, pady=4)
            row_ents.append(b_ent)
            self.linear_widgets.append(b_ent)
            self.matrix_entries.append(row_ents)

        self.setup_enter_nav()

    def setup_enter_nav(self):
        for i, widget in enumerate(self.linear_widgets):
            if i < len(self.linear_widgets) - 1:
                widget.bind("<Return>", lambda e, w=self.linear_widgets[i + 1]: w.focus_set())
            else:
                widget.bind("<Return>", lambda e: self.run_both())

    def clear_results(self):
        self.output.delete("1.0", tk.END)

    def read_matrix(self):
        try:
            rows = self.rows_var.get()
            cols = self.cols_var.get()
            A = np.zeros((rows, cols))
            b = np.zeros(rows)
            for i in range(rows):
                for j in range(cols):
                    val = self.matrix_entries[i][j].get()
                    A[i, j] = float(val) if val.strip() else 0
                val_b = self.matrix_entries[i][-1].get()
                b[i] = float(val_b) if val_b.strip() else 0
            return A, b
        except:
            messagebox.showerror("Error", "Invalid Input (Check numbers)")
            return None, None

    def run_solver(self, method):
        A, b = self.read_matrix()
        if A is None: return
        self.output.delete("1.0", tk.END)
        logger = Logger(self.output)
        try:
            textbook_solve(A.copy(), b.copy(), logger.write, method)
        except Exception as ex:
            self.output.insert(tk.END, f"Error: {ex}\n", "error")

    def run_both(self):
        A, b = self.read_matrix()
        if A is None: return
        self.output.delete("1.0", tk.END)
        logger = Logger(self.output)
        try:
            logger.write(">>> PART 1: GAUSSIAN ELIMINATION <<<\n", "header")
            textbook_solve(A.copy(), b.copy(), logger.write, "gaussian")
            logger.write("\n\n" + "#" * 60 + "\n", "sep")
            logger.write(">>> PART 2: GAUSS-JORDAN ELIMINATION <<<\n", "header")
            logger.write("#" * 60 + "\n\n", "sep")
            textbook_solve(A.copy(), b.copy(), logger.write, "gauss-jordan")
        except Exception as ex:
            self.output.insert(tk.END, f"Error: {ex}\n", "error")


if __name__ == "__main__":
    root = tk.Tk()
    app = GaussApp(root)
    root.mainloop()
