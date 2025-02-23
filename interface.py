import os
import tkinter as tk
from tkinter import messagebox, simpledialog

def select_index():
    def update_listbox(*args):
        search_term = search_var.get().lower()
        filtered_indices = [idx for idx in available_indices if search_term in idx.lower()]
        listbox.delete(0, tk.END)
        for idx in filtered_indices:
            listbox.insert(tk.END, idx)

    def on_enter(event):
        selected_index[0] = listbox.get(tk.ACTIVE)
        root.destroy()

    root = tk.Tk()
    root.title("Sélectionner un indice")
    root.geometry("600x400")

    available_indices = os.listdir('./data')

    tk.Label(root, text="Rechercher un indice:").pack()
    search_var = tk.StringVar()
    search_var.trace_add("write", update_listbox)
    search_entry = tk.Entry(root, textvariable=search_var)
    search_entry.pack()
    search_entry.bind("<Return>", on_enter)  # Binding de la touche Entrée

    index_var = tk.StringVar()
    listbox = tk.Listbox(root, listvariable=index_var)
    listbox.pack(fill=tk.BOTH, expand=True)

    for idx in available_indices:
        listbox.insert(tk.END, idx)

    selected_index = [None]
    root.mainloop()
    return selected_index[0]

def initialize_trade():
    def on_enter(event):
        try:
            sell_price_win = float(profit.get())
            sell_price_loss = float(stoploss.get())
            trade_volume = float(volume.get())
            print(f"Trade initialized: Sell Price Win={sell_price_win}, Sell Price Loss={sell_price_loss}, Volume={trade_volume}")
            trade_window.destroy()
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    trade_window = tk.Toplevel()
    trade_window.title("Ordre de trade")
    trade_window.geometry("600x400")

    tk.Label(trade_window, text="Take profit").pack()
    profit = tk.StringVar()
    tk.Entry(trade_window, textvariable=profit).pack()

    tk.Label(trade_window, text="Stop loss").pack()
    stoploss = tk.StringVar()
    tk.Entry(trade_window, textvariable=stoploss).pack()

    tk.Label(trade_window, text="Volume").pack()
    volume = tk.StringVar()

    search_entry = tk.Entry(trade_window, textvariable=volume)
    search_entry.pack()
    search_entry.bind("<Return>", on_enter)
