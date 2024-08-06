import sys
import os

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaila import Vaila

if __name__ == "__main__":
    app = Vaila()
    app.mainloop()