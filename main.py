import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk

from core.image_processor import read_image
from core.defect_detector import detect_defects, draw_detections
from config import DETECTED_DIR

def apply_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#101827")
    style.configure("Panel.TFrame", background="#172235")
    style.configure("TLabel", background="#101827", foreground="#e8eef8", font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#f4f7fb")
    style.configure("Muted.TLabel", foreground="#91a1b8")
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 9),
                    background="#2864d7", foreground="white", borderwidth=0)
    style.map("TButton", background=[("active", "#3979f0")])
    style.configure("Treeview", background="#172235", fieldbackground="#172235",
                    foreground="#e8eef8", rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", background="#24344d", foreground="#dce7f7",
                    font=("Segoe UI", 9, "bold"))
    style.configure("TEntry", fieldbackground="#24344d", foreground="#ffffff")
    style.configure("TSpinbox", fieldbackground="#24344d", foreground="#ffffff",
                    arrowsize=12)

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ứng dụng kiểm tra lỗi bo mạch PCB")
        self.root.geometry("1240x790")
        self.root.minsize(1000, 680)
        self.root.configure(bg="#101827")
        apply_style(self.root)

        self.template_path = None
        self.test_path = None
        self.current_result = None
        self.preview_image = None
        self._build_ui()

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(24, 20))
        header.pack(fill="x")
        ttk.Label(header, text="Ứng dụng kiểm tra lỗi bo mạch PCB", style="Title.TLabel").pack(anchor="w")

        body = ttk.Frame(self.root, padding=(20, 4, 20, 18))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.pack(side="left", fill="y", padx=(0, 14))
        ttk.Label(left, text="THIẾT LẬP KIỂM TRA",
                  font=("Segoe UI", 11, "bold"), background="#172235",
                  foreground="#f4f7fb").pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Chọn ảnh PCB mẫu", command=self.choose_template).pack(fill="x", pady=5)
        self.template_label = ttk.Label(left, text="Chưa chọn ảnh", style="Muted.TLabel",
                                         background="#172235", wraplength=220)
        self.template_label.pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Chọn ảnh PCB kiểm tra", command=self.choose_test).pack(fill="x", pady=5)
        self.test_label = ttk.Label(left, text="Chưa chọn ảnh", style="Muted.TLabel",
                                    background="#172235", wraplength=220)
        self.test_label.pack(anchor="w", pady=(0, 12))

        ttk.Label(left, text="Diện tích lỗi tối thiểu (px)", background="#172235").pack(anchor="w", pady=(12, 3))
        self.min_area = tk.StringVar(value="8")
        ttk.Entry(left, textvariable=self.min_area, width=12).pack(anchor="w")

        # Dung sai lệch biên: weak quyết định hình dạng vùng lỗi,
        # strong quyết định vùng nào được công nhận là lỗi (strong >= weak)
        ttk.Label(left, text="Dung sai biên (px)", background="#172235").pack(anchor="w", pady=(12, 3))
        tol_frame = ttk.Frame(left, style="Panel.TFrame")
        tol_frame.pack(anchor="w", fill="x")

        weak_col = ttk.Frame(tol_frame, style="Panel.TFrame")
        weak_col.pack(side="left", padx=(0, 14))
        ttk.Label(weak_col, text="Weak", style="Muted.TLabel",
                  background="#172235").pack(anchor="w")
        self.tol_weak = tk.StringVar(value="1")
        ttk.Spinbox(weak_col, from_=0, to=10, textvariable=self.tol_weak,
                    width=5).pack(anchor="w")

        strong_col = ttk.Frame(tol_frame, style="Panel.TFrame")
        strong_col.pack(side="left")
        ttk.Label(strong_col, text="Strong", style="Muted.TLabel",
                  background="#172235").pack(anchor="w")
        self.tol_strong = tk.StringVar(value="1")
        ttk.Spinbox(strong_col, from_=0, to=10, textvariable=self.tol_strong,
                    width=5).pack(anchor="w")

        ttk.Label(left, text="Strong ≥ Weak. Strong lớn hơn → ít nhiễu hơn\nnhưng dễ bỏ sót lỗi nhỏ.",
                  style="Muted.TLabel", background="#172235",
                  wraplength=220, font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        self.align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Tự căn chỉnh ảnh", variable=self.align_var).pack(anchor="w", pady=12)
        ttk.Button(left, text="PHÂN TÍCH ẢNH", command=self.run_detection).pack(fill="x", pady=(10, 6))
        ttk.Button(left, text="Lưu ảnh kết quả", command=self.save_result).pack(fill="x", pady=5)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # Khu vực hiển thị hai ảnh
        image_frame = ttk.Frame(right)
        image_frame.pack(fill="both", expand=True)

        # Khung ảnh REF
        ref_frame = ttk.Frame(image_frame, style="Panel.TFrame", padding=10)
        ref_frame.pack(side="left", fill="both", expand=True, padx=(0, 7))

        ttk.Label(
            ref_frame,
            text="ẢNH PCB MẪU (REF)",
            background="#172235",
            foreground="#f4f7fb",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=(0, 8))

        self.ref_preview = tk.Label(
            ref_frame,
            text="Chưa chọn ảnh REF",
            bg="#172235",
            fg="#91a1b8"
        )
        self.ref_preview.pack(fill="both", expand=True)

        # Khung ảnh TEST
        test_frame = ttk.Frame(image_frame, style="Panel.TFrame", padding=10)
        test_frame.pack(side="left", fill="both", expand=True, padx=(7, 0))

        ttk.Label(
            test_frame,
            text="ẢNH PCB KIỂM TRA (TEST)",
            background="#172235",
            foreground="#f4f7fb",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=(0, 8))

        self.test_preview = tk.Label(
            test_frame,
            text="Chưa chọn ảnh TEST",
            bg="#172235",
            fg="#91a1b8"
        )
        self.test_preview.pack(fill="both", expand=True)
        bottom = ttk.Frame(right, style="Panel.TFrame", padding=14)
        bottom.pack(fill="x", pady=(14, 0))
        self.summary = ttk.Label(bottom, text="Chưa có kết quả phân tích",
                                 background="#172235", foreground="#dce7f7")
        self.summary.pack(anchor="w", pady=(0, 8))

        columns = ("index", "type", "position", "size", "area")
        self.table = ttk.Treeview(bottom, columns=columns, show="headings", height=6)
        headings = {"index": "STT", "type": "Nhóm lỗi", "position": "Vị trí (x, y)",
                    "size": "Kích thước", "area": "Diện tích"}
        widths = {"index": 55, "type": 130, "position": 150, "size": 130, "area": 100}
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="center")
        self.table.pack(fill="x")

    def show_preview(self, image, target_label):
        if image is None:
            return

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        # Lấy kích thước khung hiển thị
        target_label.update_idletasks()
        max_w = max(target_label.winfo_width(), 300)
        max_h = max(target_label.winfo_height(), 300)

        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil)

        if not hasattr(self, "preview_refs"):
            self.preview_refs = {}

        self.preview_refs[target_label] = photo

        target_label.config(image=photo, text="")
        
    def choose_template(self):
        path = filedialog.askopenfilename(
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")]
        )

        if path:
            self.template_path = path
            self.template_label.config(text=Path(path).name)

            image = read_image(path)
            self.show_preview(image, self.ref_preview)

    def choose_test(self):
        path = filedialog.askopenfilename(
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")]
        )

        if path:
            self.test_path = path
            self.test_label.config(text=Path(path).name)

            image = read_image(path)
            self.show_preview(image, self.test_preview)

    @staticmethod
    def _read_int(var, name, minimum=0):
        """Đọc số nguyên từ ô nhập, báo lỗi rõ ràng nếu không hợp lệ."""
        try:
            value = int(var.get().strip())
        except ValueError:
            raise ValueError(f"'{name}' phải là số nguyên.")
        if value < minimum:
            raise ValueError(f"'{name}' phải >= {minimum}.")
        return value

    def run_detection(self):
        if not self.template_path or not self.test_path:
            messagebox.showwarning(
                "Thiếu dữ liệu",
                "Vui lòng chọn đủ ảnh mẫu và ảnh kiểm tra."
            )
            return

        try:
            min_area = self._read_int(self.min_area, "Diện tích tối thiểu", 0)
            tol_weak = self._read_int(self.tol_weak, "Dung sai Weak", 0)
            tol_strong = self._read_int(self.tol_strong, "Dung sai Strong", 0)
        except ValueError as exc:
            messagebox.showwarning("Giá trị không hợp lệ", str(exc))
            return

        if tol_strong < tol_weak:
            messagebox.showwarning(
                "Giá trị không hợp lệ",
                "Dung sai Strong phải lớn hơn hoặc bằng Weak."
            )
            return

        try:
            template = read_image(self.template_path)
            test = read_image(self.test_path)

            result = detect_defects(
                template,
                test,
                min_area=min_area,
                tol_weak=tol_weak,
                tol_strong=tol_strong,
                use_alignment=self.align_var.get()
            )

            result["visual"] = draw_detections(
                result["aligned"],
                result["detections"]
            )

            self.current_result = result

            # Hiển thị lại hai ảnh gốc
            self.show_preview(template, self.ref_preview)
            self.show_preview(result["visual"], self.test_preview)

            self._update_table(result["detections"])

            self.summary.config(
                text=(
                    f'Phát hiện {len(result["detections"])} vùng nghi lỗi '
                    f'(weak={tol_weak}, strong={tol_strong})'
                )
            )

        except Exception as exc:
            messagebox.showerror("Lỗi xử lý", str(exc))

    def _show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((900, 510))
        self.preview_image = ImageTk.PhotoImage(pil)
        self.preview.config(image=self.preview_image, text="")

    def _update_table(self, detections):
        for row in self.table.get_children():
            self.table.delete(row)
        for i, item in enumerate(detections, 1):
            self.table.insert("", "end", values=(
                i, item["label"], f'{item["x1"]}, {item["y1"]}',
                f'{item["width"]} × {item["height"]}', f'{item["area_px"]:.0f} px²'
            ))

    def save_result(self):
        if not self.current_result:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu.")
            return
        path = DETECTED_DIR / f"result_{Path(self.test_path).stem}.png"
        if cv2.imwrite(str(path), self.current_result["visual"]):
            messagebox.showinfo("Đã lưu", f"Kết quả: {path}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    App().run()