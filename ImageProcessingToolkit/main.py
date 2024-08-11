import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO

class ImageProcessingToolkit:
    def __init__(self, root):
        self.root = root
        self.window_width = 1280
        self.window_height = 720
        self.left_image_label = None
        self.right_image_label = None
        self.image = None
        self.model = YOLO('yolov8n-seg.pt')  # Load the YOLOv8 segmentation model

        self.create_window()
        self.create_widgets()

    def create_window(self):
        self.root.title('Image Processing Toolkit')

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        center_x = int(screen_width / 2 - self.window_width / 2)
        center_y = int(screen_height / 2 - self.window_height / 2)

        self.root.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')
        self.root.resizable(True, True)

    def create_widgets(self):
        ttk.Label(self.root, text='Image Processing Toolkit').pack(pady=10)

        ttk.Button(self.root, text='Load Image', command=self.load_image).pack(pady=10)

        self.keyword_entry = ttk.Entry(self.root)
        self.keyword_entry.pack(pady=10)

        ttk.Button(self.root, text='Apply Filter', command=self.apply_filter).pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
        )

        if file_path:
            self.image = Image.open(file_path)
            resized_image = self.image.resize((self.window_width // 2 - 40, self.window_height - 200))

            img_tk = ImageTk.PhotoImage(resized_image)

            if self.left_image_label:
                self.left_image_label.config(image=img_tk)
                self.left_image_label.image = img_tk
            else:
                self.left_image_label = ttk.Label(self.root, image=img_tk)
                self.left_image_label.image = img_tk
                self.left_image_label.pack(side=tk.LEFT, padx=10, pady=20)

            if self.right_image_label:
                self.right_image_label.config(image=img_tk)
                self.right_image_label.image = img_tk
            else:
                self.right_image_label = ttk.Label(self.root, image=img_tk)
                self.right_image_label.image = img_tk
                self.right_image_label.pack(side=tk.LEFT, padx=10, pady=20)

    def apply_filter(self):
        keyword = self.keyword_entry.get().lower()

        if self.image and keyword:
            # Perform object detection and segmentation using YOLOv8
            results = self.model(self.image)

            # Convert the image to RGBA for transparency
            mask_image = self.image.convert("RGBA")
            draw = ImageDraw.Draw(mask_image, "RGBA")

            for result in results:
                for idx, seg in enumerate(result.masks.xy):
                    if keyword in result.names[int(result.boxes[idx].cls)]:
                        # Create a transparent red mask for the segmentation area
                        mask_polygon = [(int(x), int(y)) for x, y in seg]
                        draw.polygon(mask_polygon, fill=(255, 0, 0, 100))

            # Resize the masked image and display it on the right label
            resized_mask_image = mask_image.resize((self.window_width // 2 - 40, self.window_height - 200))
            img_tk_mask = ImageTk.PhotoImage(resized_mask_image)

            self.right_image_label.config(image=img_tk_mask)
            self.right_image_label.image = img_tk_mask


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingToolkit(root)
    root.mainloop()
