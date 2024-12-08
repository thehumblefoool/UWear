import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import threading
import logging
from colorama import Fore, Style
import os
import csv
from datetime import datetime

# Set up logging with color support
class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "DEBUG": Fore.CYAN
        }
        reset = Style.RESET_ALL
        log_fmt = f"{level_color.get(record.levelname, '')}%(asctime)s - %(levelname)s - %(message)s{reset}"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Set up logging
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Load YOLOv5 model
logging.info("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"yolov5\runs\train\exp3\weights\best.pt")
logging.info("YOLOv5 model loaded successfully.")

root = tk.Tk()
root.title("UWear Dress Code Detector")
root.configure(bg='white')  # white background for contrast

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 720
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(True, True)

root.columnconfigure(0, weight=1)  
root.columnconfigure(1, weight=3)  
root.columnconfigure(2, weight=2)  
root.rowconfigure(0, weight=0)    
root.rowconfigure(1, weight=1)    
root.rowconfigure(2, weight=0)    

title_frame = ttk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', pady=(10, 0))
title_label = ttk.Label(
    title_frame,
    text="UWear Dress Code Detector",
    font=("Arial", 18, "bold"),
    background="black",
    foreground="white",
    anchor="center",
)
title_label.pack(fill='x', pady=5)

content_frame = ttk.Frame(root)
content_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', padx=10, pady=(5, 0))

content_frame.columnconfigure(0, weight=1)  
content_frame.columnconfigure(1, weight=3)  
content_frame.columnconfigure(2, weight=2)  
content_frame.rowconfigure(0, weight=1)

left_spacer = ttk.Frame(content_frame, width=100)
left_spacer.grid(row=0, column=0, sticky='nsew')

video_frame = ttk.Frame(content_frame)
video_frame.grid(row=0, column=1, sticky='nsew', padx=(10, 10))
video_frame.columnconfigure(0, weight=1)
video_frame.rowconfigure(0, weight=1)

label = ttk.Label(video_frame, borderwidth=2, relief="solid")
label.grid(row=0, column=0, sticky='nsew')

side_panel_frame = ttk.Frame(content_frame)
side_panel_frame.grid(row=0, column=2, sticky='nsew')
side_panel_frame.columnconfigure(0, weight=1)
side_panel_frame.rowconfigure(0, weight=0)  
side_panel_frame.rowconfigure(1, weight=1)  

# ----- Text at the Top of the Side Panel -----
static_text_frame = ttk.Frame(side_panel_frame)
static_text_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))

# Example text guidelines - customize as needed
guidelines_text = (
    "Dress Code Guidelines According to Handbook Section 27: Code of Conduct:\n"
    "1. Skirts/Dresses: No more than 2 inches above knee.\n"
    "2. No cropped tops.\n"
    "3. No ripped pants exposing skin 3 inches above knee.\n"
    "4. Shorts: Not more than 3 inches above kneecap.\n"
    "5. No sleeveless tops.\n"
)

guidelines_label = ttk.Label(
    static_text_frame,
    text=guidelines_text,
    font=("Arial", 12),
    wraplength=350,
    anchor="center",  # Center the text horizontally within the label
    justify="center"  # Ensure that the text is centered
)
guidelines_label.pack(expand=True, fill='both', padx=5, pady=5)

guidelines_label.pack(expand=True, fill='both', padx=5, pady=5)

detected_images_canvas = tk.Canvas(side_panel_frame, bg='white')
detected_images_scrollbar = ttk.Scrollbar(side_panel_frame, orient="vertical", command=detected_images_canvas.yview)
detected_images_scrollable_frame = ttk.Frame(detected_images_canvas)

detected_images_scrollable_frame.bind(
    "<Configure>",
    lambda e: detected_images_canvas.configure(
        scrollregion=detected_images_canvas.bbox("all")
    )
)

detected_images_canvas.create_window((0, 0), window=detected_images_scrollable_frame, anchor="nw")
detected_images_canvas.configure(yscrollcommand=detected_images_scrollbar.set)

detected_images_canvas.grid(row=1, column=0, sticky='nsew')
detected_images_scrollbar.grid(row=1, column=1, sticky='ns')

IMAGE_PATH = r"C:\Users\Caleb\Desktop\UWear\sidefeedimages"

illegal_wear_images = {}
image_filenames = {
    'Skirt': 'skirt.png',
    'cropped_top': 'cropped_top.png',
    'ripped_pants': 'ripped_pants.png',
    'shorts': 'shorts.png',
    'sleeveless': 'sleeveless.png'
}

try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.LANCZOS

for key, filename in image_filenames.items():
    image_path = os.path.join(IMAGE_PATH, filename)
    try:
        img = Image.open(image_path)
        illegal_wear_images[key] = img
    except IOError:
        logging.error(f"Error loading image for {key} from {image_path}")

photo_images = {}

notification_label = ttk.Label(
    root,
    text="Detection Results: ",
    font=("Arial", 16),
    background="black",
    foreground="white",
    anchor="center",
)
notification_label.grid(row=2, column=0, columnspan=3, sticky='nsew', pady=10)

VIOLATIONS_CSV = "violations.csv"
violation_handling = False

def handle_violation(detected_items):
    global violation_handling
    if violation_handling:
        return
    violation_handling = True
    notification_label.configure(text="Violation detected! Please position yourself for scanning within 10 seconds.")
    logging.info("Violation detected. Waiting 10 seconds for positioning.")
    root.after(10000, prompt_student_number, detected_items)

def prompt_student_number(detected_items):
    global violation_handling
    recorded_items = detected_items[:2]

    student_number = simpledialog.askstring("Student Number", "Please enter your student number:")
    if student_number:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(VIOLATIONS_CSV)
        try:
            with open(VIOLATIONS_CSV, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Student Number', 'Violations'])
                writer.writerow([timestamp, student_number, "; ".join(recorded_items)])
            logging.info(f"Violation recorded for student {student_number} at {timestamp}: {recorded_items}")
            messagebox.showinfo("Violation Recorded", "Your violation has been recorded. Thank you.")
        except Exception as e:
            logging.error(f"Failed to write to CSV: {e}")
            messagebox.showerror("Error", "Failed to record the violation. Please contact support.")
    else:
        logging.warning("No student number entered. Violation not recorded.")
        messagebox.showwarning("Input Required", "No student number entered. Violation not recorded.")

    violation_handling = False
    notification_label.configure(text="Detection Results: ")

def update_gui(image, message, detected_items):
    label.imgtk = image
    label.configure(image=image)
    notification_label.configure(text=message)
    update_side_panel(detected_items)

def update_side_panel(detected_items):
    # Clear existing widgets in the scrollable frame
    for widget in detected_images_scrollable_frame.winfo_children():
        widget.destroy()
    
    for item in detected_items:
        if item in illegal_wear_images:
            img = illegal_wear_images[item]
            max_detected_width = 400
            max_detected_height = 400
            img.thumbnail((max_detected_width, max_detected_height), resample_filter)
            
            photo = ImageTk.PhotoImage(img)

            # Create a new frame for each image
            image_frame = ttk.Frame(detected_images_scrollable_frame)
            image_frame.grid(row=detected_images_scrollable_frame.grid_size()[1], column=0, pady=10, sticky='ew')
            
            # Create and pack the photo_label centered in its frame
            photo_label = ttk.Label(image_frame, image=photo)
            photo_label.image = photo  # Keep a reference to avoid garbage collection
            photo_label.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
            
            # Create and pack the description label centered in its frame
            description = ttk.Label(image_frame, text=item.replace('_', ' ').title(), font=("Arial", 12))
            description.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
            
            # Store the photo reference
            photo_images[item] = photo

    # Ensure the parent scrollable frame stretches to fit its container
    detected_images_scrollable_frame.grid_columnconfigure(0, weight=1, uniform="center")


def video_stream():
    logging.info("Starting video stream...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logging.error("Unable to access the camera.")
        return
    logging.info("Camera activated successfully.")

    DISPLAY_WIDTH = 960
    DISPLAY_HEIGHT = 600

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read frame from camera.")
            break

        logging.debug("Processing frame...")
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        results = model(frame)
        detections = results.pandas().xyxy[0]
        message = "Detection Results: "
        detected_items = []

        for index, row in detections.iterrows():
            item = row['name']
            if item == 'Skirt':
                message += "Skirt detected (Section 27.1.2.7) "
                detected_items.append('Skirt')
            elif item in ['Trousers', 'valid_top']:
                message += f"{item} detected (legal wear). "
            elif item == 'cropped_top':
                message += "Cropped top detected (Section 27.1.2.9). "
                detected_items.append('cropped_top')
            elif item == 'ripped_pants':
                message += "Ripped pants detected (Section 27.1.2.2). "
                detected_items.append('ripped_pants')
            elif item == 'shorts':
                message += "Shorts detected (Section 27.1.2.3). "
                detected_items.append('shorts')
            elif item == 'sleeveless':
                message += "Sleeveless top detected (Section 27.1.2.4/6). "
                detected_items.append('sleeveless')

        logging.info(f"Detections: {message.strip()}")
        annotated_frame = results.render()[0]
        img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        root.after(0, update_gui, imgtk, message, detected_items)

        if detected_items and not violation_handling:
            root.after(0, handle_violation, detected_items)

    cap.release()
    logging.info("Video stream stopped.")

thread = threading.Thread(target=video_stream)
thread.daemon = True
thread.start()

root.mainloop()