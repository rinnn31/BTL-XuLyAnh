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

        ttk.Label(left, text="Ngưỡng sai khác", background="#172235").pack(anchor="w", pady=(12, 3))
        self.threshold = tk.IntVar(value=28)
        ttk.Scale(left, from_=5, to=100, variable=self.threshold, orient="horizontal").pack(fill="x")
        self.threshold_value = ttk.Label(left, text="28", background="#172235")
        self.threshold_value.pack(anchor="e")
        self.threshold.trace_add("write", lambda *_: self.threshold_value.config(text=str(self.threshold.get())))

        ttk.Label(left, text="Diện tích lỗi tối thiểu (px)", background="#172235").pack(anchor="w", pady=(12, 3))
        self.min_area = tk.StringVar(value="8")
        ttk.Entry(left, textvariable=self.min_area, width=12).pack(anchor="w")

        self.align_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Tự căn chỉnh ảnh", variable=self.align_var).pack(anchor="w", pady=12)
        ttk.Button(left, text="PHÂN TÍCH ẢNH", command=self.run_detection).pack(fill="x", pady=(10, 6))
        ttk.Button(left, text="Lưu ảnh kết quả", command=self.save_result).pack(fill="x", pady=5)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self.preview = tk.Label(right, text="Chọn ảnh mẫu và ảnh kiểm tra để bắt đầu",
                                bg="#172235", fg="#91a1b8", font=("Segoe UI", 12))
        self.preview.pack(fill="both", expand=True)

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

    def choose_template(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.template_path = path
            self.template_label.config(text=Path(path).name)

    def choose_test(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.test_path = path
            self.test_label.config(text=Path(path).name)

    def run_detection(self):
        if not self.template_path or not self.test_path:
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng chọn đủ ảnh mẫu và ảnh kiểm tra.")
            return
        try:
            template = read_image(self.template_path)
            test = read_image(self.test_path)
            result = detect_defects(template, test, threshold=self.threshold.get(),
                                    min_area=int(self.min_area.get()),
                                    use_alignment=self.align_var.get())
            result["visual"] = draw_detections(result["aligned"], result["detections"])
            self.current_result = result
            self._show_image(result["visual"])
            self._update_table(result["detections"])
            self.summary.config(text=f'Phát hiện {len(result["detections"])} vùng nghi lỗi')
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
