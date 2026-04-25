import matplotlib.pyplot as plt
import pandas as pd

try:
    from .drawsportsfields import plot_field
except (ImportError, ValueError):
    from drawsportsfields import plot_field  # ty: ignore[unresolved-import]

# 1. Carregar as marcações oficiais do campo (os 37 pontos)
field_csv = "vaila/models/soccerfield_ref3d_fifa.csv"
fdf = pd.read_csv(field_csv)

# 2. Desenhar o campo de futebol base usando o seu código
# Alteramos show_reference_points para True se quiser ver os 37 pontos vermelhos também
fig, ax = plot_field(fdf, show_reference_points=True, show_axis_values=True)

# 3. Ler o ficheiro com os pontos densos (separado por espaços)
# Como o txt não tem cabeçalho, damos os nomes x, y, z
dense_points = pd.read_csv(
    "vaila/models/pitch_points.txt", sep=" ", header=None, names=["x", "y", "z"]
)

# 4. Plotar os pontos densos por cima do campo
# Usamos s=5 para o tamanho do ponto e zorder para ficar por cima das linhas
ax.scatter(
    dense_points["x"],
    dense_points["y"],
    color="blue",
    s=5,
    zorder=3,
    label="Pontos Densos (pitch_points.txt)",
)

# 5. Mostrar o resultado
plt.legend(loc="upper right")
plt.title("Validação: Pontos Densos vs. Pontos de Referência FIFA")
plt.show()
