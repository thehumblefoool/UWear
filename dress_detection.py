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

# ------------------------------------------
# Setup Logging with Color Support
# ------------------------------------------
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

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# ------------------------------------------
# Load YOLOv5 model
# ------------------------------------------
logging.info("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"yolov5\runs\train\exp3\weights\best.pt")
logging.info("YOLOv5 model loaded successfully.")

# ------------------------------------------
# Main Application Window
# ------------------------------------------
root = tk.Tk()
root.title("UWear Dress Code Detector")

# Overall colors and styling
BG_COLOR = "#F0F0F0"
HEADER_BG = "#333333"
HEADER_FG = "white"
GUIDELINE_BG = "#FFFFFF"
SIDEPANEL_BG = "#FAFAFA"

root.configure(bg=BG_COLOR)
root.geometry("1920x720")
root.resizable(True, True)

# ------------------------------------------
# Configure Styles
# ------------------------------------------
style = ttk.Style(root)
style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground="black")
style.configure("Title.TLabel", font=("Arial", 18, "bold"), background=HEADER_BG, foreground=HEADER_FG)
style.configure("Section.TLabelframe", background=GUIDELINE_BG)
style.configure("Section.TLabelframe.Label", font=("Arial", 14, "bold"))
style.configure("Detected.TLabelframe", background=SIDEPANEL_BG)
style.configure("Detected.TLabelframe.Label", font=("Arial", 14, "bold"))

# ------------------------------------------
# Layout: 
# Top row: Title
# Middle row: Video (left), Guidelines & Detected Items (right)
# Bottom row: Status/Detection Results
# ------------------------------------------
root.rowconfigure(0, weight=0)  # title bar
root.rowconfigure(1, weight=1)  # main content
root.rowconfigure(2, weight=0)  # bottom status
root.columnconfigure(0, weight=1)

# ------------------------------------------
# Title Bar
# ------------------------------------------
title_frame = ttk.Frame(root)
title_frame.grid(row=0, column=0, sticky='nsew')
title_frame.columnconfigure(0, weight=1)

title_label = ttk.Label(
    title_frame,
    text="UWear Dress Code Detector",
    style="Title.TLabel",
    anchor="center"
)
title_label.grid(row=0, column=0, sticky='nsew', pady=5)

# ------------------------------------------
# Main Content Frame
# ------------------------------------------
main_frame = ttk.Frame(root)
main_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
main_frame.columnconfigure(0, weight=3)  # video feed
main_frame.columnconfigure(1, weight=1)  # side panel
main_frame.rowconfigure(0, weight=1)

# Video Frame (Left)
video_frame = ttk.Frame(main_frame, padding=10, style="TFrame")
video_frame.grid(row=0, column=0, sticky='nsew')
video_frame.columnconfigure(0, weight=1)
video_frame.rowconfigure(0, weight=1)

label = ttk.Label(video_frame, borderwidth=2, relief="solid")
label.grid(row=0, column=0, sticky='nsew')

# Side Panel Frame (Right)
side_panel_frame = ttk.Frame(main_frame, padding=(10,10), style="TFrame")
side_panel_frame.grid(row=0, column=1, sticky='nsew')
side_panel_frame.columnconfigure(0, weight=1)
side_panel_frame.rowconfigure(0, weight=0)  
side_panel_frame.rowconfigure(1, weight=1)  

# Guidelines Section
guidelines_frame = ttk.Labelframe(side_panel_frame, text="Dress Code Guidelines", style="Section.TLabelframe")
guidelines_frame.grid(row=0, column=0, sticky='new', padx=0, pady=(0,10))
guidelines_frame.columnconfigure(0, weight=1)

guidelines_text = (
    "According to Handbook Section 27: Code of Conduct:\n"
    "1. Skirts/Dresses: No more than 2 inches above knee.\n"
    "2. No cropped tops.\n"
    "3. No ripped pants exposing skin 3 inches above knee.\n"
    "4. Shorts: Not more than 3 inches above kneecap.\n"
    "5. No sleeveless tops."
)

guidelines_label = ttk.Label(
    guidelines_frame,
    text=guidelines_text,
    font=("Arial", 12),
    wraplength=300,
    background=GUIDELINE_BG,
    justify="center",  # Center the text
    anchor="center"
)
guidelines_label.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

# Detected Items Section
detected_frame = ttk.Labelframe(side_panel_frame, text="Detected Items", style="Detected.TLabelframe")
detected_frame.grid(row=1, column=0, sticky='nsew')
detected_frame.columnconfigure(0, weight=1)
detected_frame.rowconfigure(0, weight=1)

# Use a Canvas with a Scrollbar and a Frame inside for better alignment control
detected_canvas = tk.Canvas(detected_frame, background=SIDEPANEL_BG)
detected_canvas.grid(row=0, column=0, sticky='nsew')

detected_scrollbar = ttk.Scrollbar(detected_frame, orient='vertical', command=detected_canvas.yview)
detected_scrollbar.grid(row=0, column=1, sticky='ns')

detected_canvas.configure(yscrollcommand=detected_scrollbar.set)

# Create a frame inside the canvas
detected_items_frame = ttk.Frame(detected_canvas, style="Detected.TLabelframe")
detected_canvas.create_window((0, 0), window=detected_items_frame, anchor='nw')

def on_frame_configure(event):
    detected_canvas.configure(scrollregion=detected_canvas.bbox("all"))

detected_items_frame.bind("<Configure>", on_frame_configure)

# ------------------------------------------
# Bottom Status/Detection Results
# ------------------------------------------
status_frame = ttk.Frame(root)
status_frame.grid(row=2, column=0, sticky='nsew')
status_frame.columnconfigure(0, weight=1)

notification_label = ttk.Label(
    status_frame,
    text="Detection Results: ",
    font=("Arial", 14, "bold"),
    background=HEADER_BG,
    foreground=HEADER_FG,
    anchor="center"
)
notification_label.grid(row=0, column=0, sticky='nsew', pady=10)

# ------------------------------------------
# Violation Handling
# ------------------------------------------
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
    # Clear existing widgets in the detected_items_frame
    for widget in detected_items_frame.winfo_children():
        widget.destroy()
    
    if detected_items:
        for item in detected_items:
            item_text = f"- {item.replace('_', ' ').title()} detected."
            item_label = ttk.Label(
                detected_items_frame,
                text=item_text,
                font=("Arial", 12),
                background=SIDEPANEL_BG,
                anchor="center",
                justify="center"
            )
            item_label.pack(pady=5, padx=10, anchor='center')
    else:
        no_violation_label = ttk.Label(
            detected_items_frame,
            text="No violations detected.",
            font=("Arial", 12, "italic"),
            background=SIDEPANEL_BG,
            anchor="center",
            justify="center"
        )
        no_violation_label.pack(pady=5, padx=10, anchor='center')

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
                message += "Skirt detected (Section 27.1.2.7). "
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

# Start the video stream in a separate thread
thread = threading.Thread(target=video_stream)
thread.daemon = True
thread.start()

root.mainloop()