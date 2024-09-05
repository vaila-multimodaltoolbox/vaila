"""
vaila.py
Version: 2024-07-19 23:00:00
"""

import tkinter as tk
from PIL import Image, ImageTk
import os


def show_vaila_message():
    window = tk.Toplevel()
    window.title("vailá")
    window.geometry("900x820")

    # Load the image
    image_path = os.path.join("vaila", "images", "vaila_logo.png")
    image = Image.open(image_path)
    image = image.resize((150, 150), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    # Create and place the image label
    image_label = tk.Label(window, image=photo)
    image_label.image = photo  # Keep a reference to avoid garbage collection
    image_label.pack(pady=10)

    # Create a frame for the text and scrollbar
    text_frame = tk.Frame(window)
    text_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # Create the scrollbar
    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create the text widget
    text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure the scrollbar
    scrollbar.config(command=text_widget.yview)

    # Create and insert the text message
    message = """
    If you have new ideas or suggestions, please send them to us.
    Join us in the liberation from paid software with the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox".

    ---------------------------------------------------------
    vailá manifest!

    In front of you stands a versatile and anarcho-integrated tool, designed to challenge the boundaries of commercial systems.
    This software, not a mere substitute, is a symbol of innovation and freedom, now available and accessible.
    However, this brave visitation of an old problem is alive and determined to eliminate these venal and
    virulent barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.
    We have left the box open with vailá to insert your ideas and processing in a liberated manner.
    The only verdict is versatility; a vendetta against exorbitant costs, held as a vow, not in vain, for the value and veracity of which shall one day
    vindicate the vigilant and the virtuous in the field of motion analysis.
    Surely, this torrent of technology tends to be very innovative, so let me simply add that it is a great honor to have you with us
    and you may call this tool vailá.
     
    ---------------------------------------------------------
    ― The vailá idea!

    "vailá" é uma expressão que mistura a sonoridade da palavra francesa "voilà" com o incentivo direto em português "vai lá".
    É uma chamada à ação, um convite à iniciativa e à liberdade de explorar, experimentar e criar sem as limitações impostas por softwares comerciais caros.
    "vailá" significa "vai lá e faça!", encorajando todos a aproveitar o poder das ferramentas versáteis e integradas do "vailá: Análise versátil da libertação anarquista integrada na caixa de ferramentas multimodal" para realizar análises com dados de múltiplos sistemas.

    "vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese BR "vai lá" (pronounced "vai-lah").
    It is a call to action, an invitation to initiative and freedom to explore, experiment, and create without the constraints imposed by expensive commercial software.
    "vailá" means "go there and do it!", encouraging everyone to harness the power of the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox" to perform analysis with data from multiple systems.
    """
    text_widget.insert(tk.END, message)
    text_widget.config(state=tk.DISABLED)  # Make the text widget read-only

    window.transient()
    window.grab_set()
    window.mainloop()


if __name__ == "__main__":
    show_vaila_message()
