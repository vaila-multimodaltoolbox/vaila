import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

import cv2
import numpy as np


def calibrate_camera_from_images(calib_files, pattern_size, square_size, show_images=False):
    """
    Receives a list of paths to chessboard images,
    detects the internal corners and returns (camera_matrix, dist_coeffs).
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size

    # Criar diretório para salvar resultados da calibração
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(calib_files[0]), f"calibration_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Diretório para salvar imagens detectadas
    detection_dir = os.path.join(output_dir, "detection_images")
    os.makedirs(detection_dir, exist_ok=True)

    objpoints, imgpoints = [], []
    for idx, f in enumerate(calib_files):
        img = cv2.imread(f)
        if img is None:
            print(f"[WARN] could not read {f}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            print(f"[WARN] corners not found in {os.path.basename(f)}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"[OK] corners detected in {os.path.basename(f)}")

        # Desenhar e salvar imagem com corners (mas não mostrar)
        detection_img = img.copy()
        cv2.drawChessboardCorners(detection_img, pattern_size, corners, found)
        base_name = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(detection_dir, f"{base_name}_detected.jpg"), detection_img)

        # Mostrar apenas se solicitado
        if show_images:
            cv2.imshow("Detection", detection_img)
            cv2.waitKey(100)

    if not objpoints:
        raise RuntimeError("No valid calibration images found.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        np.zeros((3, 3), dtype=np.float32),
        np.zeros(5, dtype=np.float32),
    )

    # Analisar causas de erro alto
    reprojection_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reprojection_errors.append((os.path.basename(calib_files[i]), error))

    # Ordenar pelos erros (maiores primeiro)
    reprojection_errors.sort(key=lambda x: x[1], reverse=True)

    print("\n Calibration completed.")
    print("Intrinsic matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel(), "\n")
    print(f"Calibration RMS error: {ret}")

    # Salvar informações de calibração
    with open(os.path.join(output_dir, "calibration_info.txt"), "w") as f:
        f.write("Calibration Information\n")
        f.write("======================\n\n")
        f.write(f"Pattern size: {pattern_size}\n")
        f.write(f"Square size: {square_size} cm\n")
        f.write(f"Number of images used: {len(objpoints)}\n")
        f.write(f"RMS error: {ret}\n\n")
        f.write("Intrinsic matrix:\n")
        f.write(f"{mtx}\n\n")
        f.write("Distortion coefficients:\n")
        f.write(f"{dist.ravel()}\n\n")
        f.write("Images with highest reprojection errors:\n")
        for img_name, error in reprojection_errors[:5]:
            f.write(f"  {img_name}: {error:.4f}\n")

    # Salvar parâmetros nos formatos requeridos
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()

    with open(os.path.join(output_dir, "camera_parameters.csv"), "w") as f:
        f.write("fx,fy,cx,cy,k1,k2,p1,p2,k3\n")
        f.write(
            f"{fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f},{k1:.17f},{k2:.17f},{p1:.17f},{p2:.17f},{k3:.17f}\n"
        )

    # Verificar e alertar sobre possíveis problemas
    if ret > 1.0:
        print("\n[WARN] O erro de calibração (RMS) é relativamente alto.")
        print("Possíveis causas:")
        print("1. Tabuleiro não está totalmente plano em algumas imagens")
        print("2. Movimento/desfoque durante a captura")
        print("3. Número insuficiente de posições diversas do tabuleiro")
        print("4. Tamanho do quadrado incorreto (não é realmente 10cm)")
        print("5. Baixa resolução das imagens ou iluminação inadequada")
        print("\nSugestões: Tente remover as imagens com maior erro de projeção:")
        for img_name, error in reprojection_errors[:3]:
            print(f"  - {img_name}: {error:.4f}")

    return mtx, dist, output_dir


def undistort_image(img_path, camera_matrix, dist_coeffs, alpha, output_dir=None):
    """
    Processes an image with different alpha values and saves all results.
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Criar diretório para resultados se não fornecido
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(img_path), f"undistorted_{timestamp}")

    # Criar subdiretórios
    os.makedirs(output_dir, exist_ok=True)

    # Processar com diferentes valores de alpha (inclui o valor especificado pelo usuário)
    alpha_values = [0, 0.2, 0.5, 0.7, 1.0]
    if alpha not in alpha_values:
        alpha_values.append(alpha)
        alpha_values.sort()

    # Processar cada alpha e salvar resultado
    results = {}
    for a in alpha_values:
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), a, (w, h))
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_mtx)

        # Aplicar recorte se ROI for válido
        if roi[2] > 0 and roi[3] > 0:
            x, y, w_roi, h_roi = roi
            undistorted_roi = undistorted[y : y + h_roi, x : x + w_roi]
        else:
            undistorted_roi = undistorted

        results[a] = undistorted

        # Salvar imagem
        base, ext = os.path.splitext(os.path.basename(img_path))
        out_name = f"{base}_alpha_{a:.1f}{ext}"
        output_path = os.path.join(output_dir, out_name)
        cv2.imwrite(output_path, undistorted)

    # Criar visão comparativa
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    compare_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")

    # Preparar imagens para comparação
    thumbs = [cv2.resize(img, (w // 3, h // 3))]  # Original
    for a in alpha_values:
        resized = cv2.resize(results[a], (w // 3, h // 3))
        cv2.putText(
            resized,
            f"Alpha: {a:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        thumbs.append(resized)

    # Organizar em grid (2 linhas x 3 colunas)
    row1 = np.hstack(thumbs[:3])
    row2 = np.hstack(
        [np.zeros_like(thumbs[0]) if len(thumbs) <= 3 else thumbs[3]]
        + [np.zeros_like(thumbs[0]) if len(thumbs) <= 4 else thumbs[4]]
        + [np.zeros_like(thumbs[0]) if len(thumbs) <= 5 else thumbs[5]]
    )
    comparison = np.vstack([row1, row2])
    cv2.imwrite(compare_path, comparison)

    # Salvar parâmetros
    fs = cv2.FileStorage(os.path.join(output_dir, f"{base_name}_calib.xml"), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("alpha", alpha)
    fs.release()

    # Salvar em formato CSV para vaila_datdistort.py
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    with open(os.path.join(output_dir, f"{base_name}_calib.csv"), "w") as f:
        f.write("fx,fy,cx,cy,k1,k2,p1,p2,k3\n")
        f.write(
            f"{fx:.2f},{fy:.2f},{cx:.2f},{cy:.2f},{k1:.17f},{k2:.17f},{p1:.17f},{p2:.17f},{k3:.17f}\n"
        )

    # Mostrar comparação (automaticamente ajustada à tela)
    show_comparison = comparison.copy()
    # Get screen dimensions without using self
    screen_width = 1280
    screen_height = 720

    # Ajustar à tela
    scale = min(
        screen_width / show_comparison.shape[1],
        screen_height / show_comparison.shape[0],
    )
    if scale < 1:
        show_comparison = cv2.resize(show_comparison, None, fx=scale, fy=scale)

    cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Comparison", show_comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    messagebox.showinfo(
        "Saved", f"All results saved to:\n{output_dir}\n\nBest alpha value: {alpha}"
    )
    return output_dir


def analyze_chessboard(calib_files, pattern_size):
    """
    Analyses the dimensions of the board in the images
    """
    for f in calib_files:
        img = cv2.imread(f)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if found:
            # Calculate the distance between adjacent corners (estimate of the square size)
            pixel_sizes = []
            for i in range(len(corners) - 1):
                if i % pattern_size[0] != pattern_size[0] - 1:  # Do not calculate between lines
                    dist = np.sqrt(
                        (corners[i + 1][0][0] - corners[i][0][0]) ** 2
                        + (corners[i + 1][0][1] - corners[i][0][1]) ** 2
                    )
                    pixel_sizes.append(dist)

            avg_size = np.mean(pixel_sizes)
            print(
                f"{os.path.basename(f)}: Found {len(corners)} corners, average size: {avg_size:.1f} pixels"
            )

            # Show image with detected corners
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            cv2.putText(
                img,
                f"Corners: {len(corners)}, Size: {avg_size:.1f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            show_resized_image("Analysis", img)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()


def enhanced_chessboard_detection(calib_files, pattern_sizes=None):
    """
    Improved function to detect boards with multiple patterns and methods.
    """
    if pattern_sizes is None:
        # Main pattern sizes is default is 8x6
        pattern_sizes = [(8, 6), (6, 8), (7, 7), (8, 8), (6, 6), (9, 9)]

    results = {}

    for f in calib_files:
        img = cv2.imread(f)
        if img is None:
            print(f"[ERROR] Could not read: {f}")
            continue

        img_name = os.path.basename(f)
        print(f"\n[INFO] Processing: {img_name}")

        # Prepare image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to improve the contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Show images side by side
        compare = np.hstack([gray, enhanced])
        show_resized_image("Original vs Enhanced", compare)
        cv2.waitKey(500)

        best_success = False
        best_corners = None
        best_pattern = None
        best_method = None

        # Methods to try
        methods = [
            (
                "Normal",
                gray,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            ),
            (
                "Enhanced",
                enhanced,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            ),
            (
                "FILTER_QUADS",
                gray,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS,
            ),
            (
                "Enhanced+FILTER_QUADS",
                enhanced,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS,
            ),
            (
                "SimpleBLOB",
                gray,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FILTER_QUADS,
            ),
        ]

        # Try to find the board with different patterns and methods
        for method_name, img_to_use, flags in methods:
            for pattern in pattern_sizes:
                found, corners = cv2.findChessboardCorners(img_to_use, pattern, flags=flags)

                if found:
                    print(f"[SUCCESS] Found with {method_name}, pattern={pattern}")
                    best_success = True
                    best_corners = corners
                    best_pattern = pattern
                    best_method = method_name
                    break

            if best_success:
                break

        # Show results
        result_img = img.copy()
        if best_success and best_corners is not None and best_pattern is not None:
            # Refine found corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corner_count = 0

            try:
                refined_corners = cv2.cornerSubPix(gray, best_corners, (11, 11), (-1, -1), criteria)
                corner_count = len(refined_corners)
            except Exception as e:
                print(f"[ERROR] cornerSubPix falhou: {e}")
                refined_corners = best_corners
                corner_count = len(best_corners)

            try:
                pattern_size = (
                    int(best_pattern[0]),
                    int(best_pattern[1]),
                )  # Ensure it is an integer tuple
                cv2.drawChessboardCorners(result_img, pattern_size, refined_corners, True)
            except Exception as e:
                print(f"[ERROR] drawChessboardCorners failed: {e}")

            results[img_name] = {
                "pattern": best_pattern,
                "method": best_method,
                "corners": corner_count,
            }

            cv2.putText(
                result_img,
                f"DETECTED: pattern={best_pattern}, method={best_method}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                result_img,
                "NOT DETECTED with any method/pattern",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            print(f"[FAIL] Could not detect in: {img_name}")

            # Try with findCirclesGrid as alternative
            try:
                circle_pattern = (4, 11)
                # findCirclesGrid needs only the pattern and optional flags
                found_circles, corners_circles = cv2.findCirclesGrid(
                    gray, circle_pattern, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
                )

                if found_circles and corners_circles is not None:
                    print("[INFO] Detected as asymmetric circle grid!")
                    try:
                        cv2.drawChessboardCorners(result_img, circle_pattern, corners_circles, True)
                    except Exception as e:
                        print(f"[ERROR] drawChessboardCorners for circles failed: {e}")
            except Exception as e:
                print(f"[ERROR] findCirclesGrid failed: {e}")
                found_circles = False

        show_resized_image("Final Result", result_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()
    return results


def show_resized_image(title, img, max_width=1024, max_height=768):
    """Helper to display images with controlled size, but larger than before"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)

    if scale < 1:  # Only resize if image is too large
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        cv2.imshow(title, img_resized)
    else:
        cv2.imshow(title, img)

    # Configure window to be resizable by the user
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calibration and Undistort")
        self.resizable(True, True)  # Permitir redimensionamento

        # Determinar tamanho da tela
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(600, screen_width - 100)
        window_height = min(400, screen_height - 100)

        # Posicionar a janela no centro
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Frame principal
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Internal corners (for 4x5 black squares: 3x4 internal corners)
        row = 0
        tk.Label(main_frame, text="Internal corners (cols, rows):").grid(
            row=row, column=0, sticky="w"
        )
        entry_frame = tk.Frame(main_frame)
        entry_frame.grid(row=row, column=1, sticky="we")
        self.cols = tk.Entry(entry_frame, width=5)
        self.cols.insert(0, "8")
        self.cols.pack(side=tk.LEFT, padx=(0, 5))
        self.rows = tk.Entry(entry_frame, width=5)
        self.rows.insert(0, "6")
        self.rows.pack(side=tk.LEFT)

        # Square size
        row += 1
        tk.Label(main_frame, text="Square size (cm):").grid(row=row, column=0, sticky="w")
        self.size = tk.Entry(main_frame)
        self.size.insert(0, "10.0")
        self.size.grid(row=row, column=1, sticky="we")

        # Alpha
        row += 1
        tk.Label(main_frame, text="Alpha [0–1]:").grid(row=row, column=0, sticky="w")
        self.alpha = tk.Entry(main_frame)
        self.alpha.insert(0, "0.5")
        self.alpha.grid(row=row, column=1, sticky="we")

        # Show images during calibration
        row += 1
        self.show_images = tk.BooleanVar(value=False)
        tk.Checkbutton(
            main_frame, text="Show images during calibration", variable=self.show_images
        ).grid(row=row, column=0, columnspan=2, sticky="w")

        # Preset parameters
        row += 1
        self.use_preset = tk.BooleanVar(value=False)
        tk.Checkbutton(
            main_frame, text="Use preset camera parameters", variable=self.use_preset
        ).grid(row=row, column=0, columnspan=2, sticky="w")

        # Buttons
        row += 1
        buttons_frame = tk.Frame(main_frame)
        buttons_frame.grid(row=row, column=0, columnspan=2, sticky="we", pady=10)

        tk.Button(buttons_frame, text="Select calibration", command=self.load_calib, width=15).pack(
            side=tk.LEFT, padx=5, expand=True, fill=tk.X
        )
        tk.Button(buttons_frame, text="Process image", command=self.run_undistort, width=15).pack(
            side=tk.LEFT, padx=5, expand=True, fill=tk.X
        )

        row += 1
        buttons_frame2 = tk.Frame(main_frame)
        buttons_frame2.grid(row=row, column=0, columnspan=2, sticky="we")

        tk.Button(buttons_frame2, text="Analyze board", command=self.analyze_board, width=15).pack(
            side=tk.LEFT, padx=5, expand=True, fill=tk.X
        )
        tk.Button(
            buttons_frame2,
            text="Debug Detection",
            command=self.debug_detection,
            bg="orange",
            width=15,
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.calib_files = []

        # Configurar redimensionamento
        main_frame.columnconfigure(1, weight=1)

    def load_calib(self):
        files = filedialog.askopenfilenames(
            title="Select calibration images (png/jpg)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if files:
            self.calib_files = list(files)
            messagebox.showinfo("OK", f"{len(files)} calibration images loaded")

    def run_undistort(self):
        if not self.calib_files:
            messagebox.showwarning("Error", "Load calibration images first.")
            return

        alpha = float(self.alpha.get())

        if self.use_preset.get():
            # Usar valores predefinidos
            fx, fy = 949.41, 950.63
            cx, cy = 960.00, 540.00
            k1, k2, k3 = -0.28871370110181493, 0.1374614711665278, -0.025511562284832402
            p1, p2 = 0.00044281215436799446, -0.00042111749309847274

            mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        else:
            try:
                pattern = (int(self.cols.get()), int(self.rows.get()))
                square = float(self.size.get())
                mtx, dist, calib_dir = calibrate_camera_from_images(
                    self.calib_files, pattern, square, self.show_images.get()
                )
            except Exception as e:
                messagebox.showerror("Calibration failed", str(e))
                return

        # Selecionar imagem para processar
        img = filedialog.askopenfilename(
            title="Select image to undistort",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")],
        )

        if img:
            output_dir = os.path.join(os.path.dirname(img), "undistorted_results")
            undistort_image(img, mtx, dist, alpha, output_dir)

    def analyze_board(self):
        if not self.calib_files:
            messagebox.showwarning("Error", "Load calibration images first.")
            return
        pattern = (int(self.cols.get()), int(self.rows.get()))
        analyze_chessboard(self.calib_files, pattern)

    def debug_detection(self):
        if not self.calib_files:
            messagebox.showwarning("Error", "Load calibration images first.")
            return

        # Add more values of pattern_size to try
        pattern_sizes = [
            (int(self.cols.get()), int(self.rows.get())),
            (int(self.cols.get()) + 1, int(self.rows.get())),
            (int(self.cols.get()) - 1, int(self.rows.get())),
            (int(self.cols.get()), int(self.rows.get()) + 1),
            (int(self.cols.get()), int(self.rows.get()) - 1),
        ]

        results = enhanced_chessboard_detection(self.calib_files, pattern_sizes)

        if results:
            # Show results in a window
            result_text = "Detection results:\n\n"
            for img, data in results.items():
                result_text += f"{img}: pattern={data['pattern']}, method={data['method']}\n"

            messagebox.showinfo("Detection results", result_text)
        else:
            messagebox.showinfo("Results", "No board detected.")


if __name__ == "__main__":
    App().mainloop()
