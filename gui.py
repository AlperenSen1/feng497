from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from predict_file import analyze_file

ROOT = Path(__file__).resolve().parent

root = tk.Tk()
root.title("Rulman Arıza Kontrolü")

folder_icon = tk.PhotoImage(file=ROOT / "assets" / "open-folder.png")

selected_file_path = None  # analizde kullanılacak dosya yolu

tk.Label(root, text="Select a file:").grid(row=0, column=0, padx=8, pady=8, sticky="w")

file_entry = tk.Entry(root, width=35, bg="white", state="readonly")  # dosya seçildikten sonra belireceği alan
file_entry.grid(row=0, column=1, padx=4, pady=8)


def open_file():
    global selected_file_path
    filepath = filedialog.askopenfilename(
        initialdir="C:/",
        title="Select A File",
        filetypes=[("MAT files", "*.mat")],
    )

    if filepath:
        selected_file_path = filepath
        file_entry.config(state="normal")
        file_entry.delete(0, tk.END)
        file_entry.insert(0, Path(filepath).name)
        file_entry.config(state="readonly")

def analyze(path):
    pass



#programın tam ortada açılması için
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w // 2) - (width // 2)
    y = (screen_h // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")

my_btn = tk.Button(root, image=folder_icon, command=open_file)
my_btn.grid(row=0, column=2, padx=4, pady=8)

analyze_btn = tk.Button(
    root,
    text="Analyze File",
    command=lambda: analyze(selected_file_path),
    bg="#008CBA",
    fg="white",
    activebackground="#1e4e8c",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    width=16,
    height=2,
    relief="flat",
)
analyze_btn.grid(row=1, column=1, padx=4, pady=8)


center(root)
root.mainloop()