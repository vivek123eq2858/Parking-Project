import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import easyocr
import threading

class NumberPlateApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.video_source = video_source

        # Open video source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Unable to open video source")
            self.window.destroy()
            return

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Canvas for video frames
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Label to show recognized plate text
        self.text_label = tk.Label(window, text="Detected Number Plate: None", font=('Helvetica', 14))
        self.text_label.pack(pady=10)

        # Button to close app
        self.btn_quit = tk.Button(window, text="Quit", command=self.on_closing)
        self.btn_quit.pack(pady=5)

        # Initialize EasyOCR reader once for performance
        self.reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if GPU available

        # Lock for OCR threading
        self.lock = threading.Lock()
        self.detected_text = None

        # Start loop
        self.delay = 15  # in ms (~60fps)
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # Detect number plate rectangles
            plates = self.detect_plates(frame)

            # Draw rectangles and OCR results
            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop ROI for OCR
                roi = frame[y:y+h, x:x+w]

                # Perform OCR in a separate thread to keep UI responsive
                if not hasattr(self, 'ocr_thread') or not self.ocr_thread.is_alive():
                    self.ocr_thread = threading.Thread(target=self.perform_ocr, args=(roi,))
                    self.ocr_thread.start()

            # Convert the image for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            # Update the detected text label thread-safely
            self.lock.acquire()
            if self.detected_text:
                display_text = f"Detected Number Plate: {self.detected_text}"
            else:
                display_text = "Detected Number Plate: None"
            self.lock.release()

            self.text_label.config(text=display_text)

        self.window.after(self.delay, self.update)

    def detect_plates(self, frame):
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Edge detection
        edged = cv2.Canny(blur, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6 and 15 < h < 150:  # Typical plate aspect ratio and size bounds
                plates.append((x, y, w, h))
        return plates

    def perform_ocr(self, roi):
        # OCR on cropped plate image
        # Convert BGR to RGB
        plate_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(plate_img, detail=0)
        text = " ".join(results).strip()
        # Save text thread-safely
        self.lock.acquire()
        if text:
            self.detected_text = text
        else:
            self.detected_text = None
        self.lock.release()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            # Release camera resource
            if self.vid.isOpened():
                self.vid.release()
            self.window.destroy()

if __name__ == "__main__":
    NumberPlateApp(tk.Tk(), "Real-Time Number Plate Detection")
