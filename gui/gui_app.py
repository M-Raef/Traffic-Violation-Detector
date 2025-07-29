"""
Updated GUI Application for Traffic Violation Detector.

This application now includes:
1. Setup tab for configuration
2. Results tab for detection results
3. Violations tab for processing/reviewing violations
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import json
from datetime import datetime
import sqlite3

# Import configuration
import config

# Import modules
from gui.roi_selector import ROISelector
from main import run_from_gui
from violation_processor import ViolationProcessor

class TrafficViolationGUI:
    """
    Main GUI application for the traffic violation detector.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Violation Detector")
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # Store a reference to this GUI instance in the root
        root.gui_app = self
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_path = tk.StringVar(value=config.YOLO_MODEL_PATH)
        self.violation_type = tk.StringVar(value="redlight")
        self.roi_data = None
        self.speed_limit = tk.DoubleVar(value=30.0)
        self.distance_meters = tk.DoubleVar(value=5.0)
        self.plate_reading_interval = tk.IntVar(value=config.PLATE_READING_INTERVAL)
        self.detector_thread = None
        self.processing = False
        
        # Violation processor
        self.violation_processor = ViolationProcessor(db_path=config.DB_PATH)
        
        # Create the main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the tabs
        self.tab_control = ttk.Notebook(main_frame)
        
        # Setup tab
        setup_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(setup_tab, text="Setup")
        self.create_setup_tab(setup_tab)
        
        # Results tab
        results_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(results_tab, text="Results")
        self.create_results_tab(results_tab)
        
        # Violations tab
        violations_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(violations_tab, text="Violations")
        self.create_violations_tab(violations_tab)
        
        # Add tabs to main frame
        self.tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Initialize results display
        self.violation_count = 0
        self.results_text = None

    def create_setup_tab(self, parent):
        """Create the setup tab with all the configuration options."""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        # Video input
        ttk.Label(file_frame, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.video_path, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_video).grid(row=0, column=2, pady=5)
        
        # Output file
        ttk.Label(file_frame, text="Output Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_path, width=50).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=5)
        
        # Model selection
        ttk.Label(file_frame, text="YOLO Model:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.model_path, width=50).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_model).grid(row=2, column=2, pady=5)
        
        file_frame.columnconfigure(1, weight=1)
        
        # Violation type section
        violation_frame = ttk.LabelFrame(frame, text="Violation Type", padding="10")
        violation_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(violation_frame, text="Red Light Violation", variable=self.violation_type, value="redlight").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(violation_frame, text="Speed Violation", variable=self.violation_type, value="speed").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ROI Selection section
        roi_frame = ttk.LabelFrame(frame, text="Region of Interest", padding="10")
        roi_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(roi_frame, text="Select ROI", command=self.select_roi).grid(row=0, column=0, pady=5)
        self.roi_status = ttk.Label(roi_frame, text="No ROI selected")
        self.roi_status.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Speed parameters section (only visible when speed violation is selected)
        self.speed_frame = ttk.LabelFrame(frame, text="Speed Parameters", padding="10")
        
        ttk.Label(self.speed_frame, text="Speed Limit (km/h):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.speed_frame, from_=5, to=200, textvariable=self.speed_limit, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.speed_frame, text="Distance between lines (meters):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.speed_frame, from_=1, to=100, textvariable=self.distance_meters, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Update speed frame visibility based on violation type
        self.violation_type.trace("w", self.update_gui_for_violation_type)
        self.update_gui_for_violation_type()
        
        # Run button at the bottom
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Detection", command=self.run_detection, style='Accent.TButton')
        self.run_button.pack(side=tk.RIGHT)
        
        # Style for accent button
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

    def create_results_tab(self, parent):
        """Create the results tab to display detection results."""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Results display
        results_frame = ttk.LabelFrame(frame, text="Detection Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Text widget for results
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Initialize with placeholder text
        self.results_text.insert(tk.END, "Run detection to see results here...")
        self.results_text.config(state=tk.DISABLED)
        
        # Button to open results folder
        ttk.Button(frame, text="Open Results Folder", command=self.open_results_folder).pack(side=tk.RIGHT, pady=5)

    def create_violations_tab(self, parent):
        """Create the violations tab for processing and reviewing violations."""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.LabelFrame(frame, text="Violation Processing", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Process pending violations button
        self.process_pending_btn = ttk.Button(
            control_frame, 
            text="Process Pending Violations",
            command=self.process_pending_violations
        )
        self.process_pending_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Refresh violations list button
        self.refresh_btn = ttk.Button(
            control_frame,
            text="Refresh List",
            command=self.refresh_violations_list
        )
        self.refresh_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Process directory button
        self.process_dir_btn = ttk.Button(
            control_frame,
            text="Process Image Directory",
            command=self.process_directory
        )
        self.process_dir_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Statistics
        self.stats_label = ttk.Label(control_frame, text="")
        self.stats_label.grid(row=0, column=3, padx=20, pady=5)
        
        # Violations list
        list_frame = ttk.LabelFrame(frame, text="Violations List", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create Treeview for violations
        columns = ('ID', 'Type', 'Plate', 'Confidence', 'Speed', 'Time', 'Status')
        self.violations_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # Define column headings
        self.violations_tree.heading('ID', text='ID')
        self.violations_tree.heading('Type', text='Type')
        self.violations_tree.heading('Plate', text='License Plate')
        self.violations_tree.heading('Confidence', text='Confidence')
        self.violations_tree.heading('Speed', text='Speed (km/h)')
        self.violations_tree.heading('Time', text='Timestamp')
        self.violations_tree.heading('Status', text='Status')
        
        # Column widths
        self.violations_tree.column('ID', width=50)
        self.violations_tree.column('Type', width=100)
        self.violations_tree.column('Plate', width=150)
        self.violations_tree.column('Confidence', width=100)
        self.violations_tree.column('Speed', width=100)
        self.violations_tree.column('Time', width=150)
        self.violations_tree.column('Status', width=100)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.violations_tree.yview)
        self.violations_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack treeview and scrollbar
        self.violations_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to view image
        self.violations_tree.bind('<Double-Button-1>', self.view_violation_image)
        
        # Actions frame
        actions_frame = ttk.Frame(frame)
        actions_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(actions_frame, text="Double-click a violation to view its image").pack(side=tk.LEFT)
        
        # Reprocess single violation button
        self.reprocess_btn = ttk.Button(
            actions_frame,
            text="Reprocess Selected",
            command=self.reprocess_selected_violation
        )
        self.reprocess_btn.pack(side=tk.RIGHT, padx=5)
        
        # Initial load
        self.refresh_violations_list()

        
#*------------------------------------------------------------------

    def process_pending_violations(self):
        """Process all pending violations for license plates."""
        self.status_var.set("Processing pending violations...")
        self.process_pending_btn.config(state=tk.DISABLED)
        
        def process():
            try:
                processed, successful = self.violation_processor.process_pending_violations()
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.status_var.set(
                    f"Processed {processed} violations, {successful} successful"
                ))
                self.root.after(0, self.refresh_violations_list)
                
                # Show completion message
                self.root.after(0, lambda: messagebox.showinfo(
                    "Processing Complete",
                    f"Processed {processed} violations\n{successful} plates extracted successfully"
                ))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.process_pending_btn.config(state=tk.NORMAL))
        
        # Run in thread
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def refresh_violations_list(self):
        """Refresh the violations list from database."""
        # Clear existing items
        for item in self.violations_tree.get_children():
            self.violations_tree.delete(item)
        
        try:
            # Get violations from database
            violations = self.violation_processor.db_manager.get_all_violations(limit=500)
            
            pending_count = 0
            total_count = len(violations)
            
            for v in violations:
                # Determine status
                if v['license_plate'].startswith('PENDING_'):
                    status = 'Pending'
                    pending_count += 1
                else:
                    status = 'Processed'
                
                # Format confidence
                conf_text = f"{v['confidence']*100:.1f}%" if v['confidence'] else "N/A"
                
                # Format speed
                speed_text = f"{v['speed']:.1f}" if v['speed'] else "N/A"
                
                # Insert into treeview
                self.violations_tree.insert('', 'end', values=(
                    v['id'],
                    v['violation_type'],
                    v['license_plate'],
                    conf_text,
                    speed_text,
                    v['timestamp'],
                    status
                ))
            
            # Update statistics
            self.stats_label.config(text=f"Total: {total_count} | Pending: {pending_count}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load violations: {str(e)}")

    def view_violation_image(self, event):
        """View the image of the selected violation."""
        selection = self.violations_tree.selection()
        if not selection:
            return
        
        item = self.violations_tree.item(selection[0])
        violation_id = item['values'][0]
        
        try:
            # Get image path from database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT image_path FROM violations WHERE id = ?", (violation_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                image_path = result[0]
                if os.path.exists(image_path):
                    # Open image with default viewer
                    if sys.platform == 'win32':
                        os.startfile(image_path)
                    elif sys.platform == 'darwin':
                        os.system(f'open "{image_path}"')
                    else:
                        os.system(f'xdg-open "{image_path}"')
                else:
                    messagebox.showwarning("Warning", "Image file not found")
            else:
                messagebox.showwarning("Warning", "No image path for this violation")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def reprocess_selected_violation(self):
        """Reprocess the selected violation."""
        selection = self.violations_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a violation to reprocess")
            return
        
        item = self.violations_tree.item(selection[0])
        violation_id = item['values'][0]
        
        self.status_var.set(f"Reprocessing violation {violation_id}...")
        
        def reprocess():
            try:
                success = self.violation_processor.reprocess_violation(violation_id)
                
                if success:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Success", f"Successfully reprocessed violation {violation_id}"
                    ))
                else:
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Failed", f"Could not extract plate for violation {violation_id}"
                    ))
                
                self.root.after(0, self.refresh_violations_list)
                self.root.after(0, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        # Run in thread
        thread = threading.Thread(target=reprocess)
        thread.daemon = True
        thread.start()

    def process_directory(self):
        """Process all images in a selected directory."""
        directory = filedialog.askdirectory(title="Select Directory with Violation Images")
        if not directory:
            return
        
        self.status_var.set(f"Processing directory: {directory}")
        
        def process():
            try:
                results = self.violation_processor.process_directory(directory)
                
                self.root.after(0, lambda: messagebox.showinfo(
                    "Processing Complete",
                    f"Processed {len(results)} images\nResults saved to plate_results.json"
                ))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        # Run in thread
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    # Keep all the original methods from your GUI
    def update_gui_for_violation_type(self, *args):
        """Update GUI elements based on the selected violation type."""
        if self.violation_type.get() == "speed":
            self.speed_frame.pack(fill=tk.X, pady=5, after=self.speed_frame.master.children["!labelframe3"])
        else:
            if self.speed_frame.winfo_ismapped():
                self.speed_frame.pack_forget()
        
        # Update ROI status if we have ROI data
        self.update_roi_status()
    
    def update_roi_status(self):
        """Update the ROI status label based on the current ROI data."""
        if self.roi_data is None:
            self.roi_status.config(text="No ROI selected")
            return
            
        if self.roi_data["type"] != self.violation_type.get():
            self.roi_data = None
            self.roi_status.config(text="No ROI selected for this violation type")
            return
            
        if self.roi_data["type"] == "redlight":
            self.roi_status.config(text=f"Red Light ROI: {len(self.roi_data['points'])} points defined")
        else:
            entry_defined = len(self.roi_data.get("entry_line", [])) == 2
            exit_defined = len(self.roi_data.get("exit_line", [])) == 2
            
            if entry_defined and exit_defined:
                self.roi_status.config(text="Speed ROI: Entry and Exit lines defined")
            elif entry_defined:
                self.roi_status.config(text="Speed ROI: Only Entry line defined")
            elif exit_defined:
                self.roi_status.config(text="Speed ROI: Only Exit line defined")
            else:
                self.roi_status.config(text="Speed ROI: Incomplete definition")

    def browse_video(self):
        """Open file dialog to select input video."""
        file_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path.set(file_path)
            # Auto-generate output path
            if not self.output_path.get():
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                self.output_path.set(os.path.join(os.path.dirname(file_path), f"{base_name}_output.avi"))

    def browse_output(self):
        """Open file dialog to select output video path."""
        file_path = filedialog.asksaveasfilename(
            title="Save Output Video As",
            filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")],
            defaultextension=".avi"
        )
        if file_path:
            self.output_path.set(file_path)

    def browse_model(self):
        """Open file dialog to select YOLO model."""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path.set(file_path)

    def select_roi(self):
        """Open ROI selection interface."""
        # Check if video file exists
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file first.")
            return
            
        # Open the video and grab a frame for ROI selection
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video: {video_path}")
            return
            
        # Try to grab a frame from middle of the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, frame_count // 2))
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            messagebox.showerror("Error", "Failed to read frame from video")
            return
            
        # Open ROI selector with the frame
        roi_selector = ROISelector(self.root, frame, roi_type=self.violation_type.get())
    
    def set_roi_result(self, result):
        """Set the ROI data from the selector."""
        self.roi_data = result
        self.update_roi_status()

    def validate_inputs(self):
        """Validate all inputs before running detection."""
        # Check video file
        if not self.video_path.get() or not os.path.exists(self.video_path.get()):
            messagebox.showerror("Error", "Please select a valid input video file.")
            return False
            
        # Check output path
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output video path.")
            return False
            
        # Check model path
        if not self.model_path.get() or not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Please select a valid YOLO model file.")
            return False
            
        # Check ROI data
        if not self.roi_data:
            messagebox.showerror("Error", "Please define a Region of Interest (ROI) first.")
            return False
            
        # Additional validation for speed detection
        if self.violation_type.get() == "speed":
            if self.roi_data["type"] != "speed":
                messagebox.showerror("Error", "Please define a Region of Interest for speed detection.")
                return False
                
            if len(self.roi_data.get("entry_line", [])) != 2 or len(self.roi_data.get("exit_line", [])) != 2:
                messagebox.showerror("Error", "Both Entry and Exit lines must be defined for speed detection.")
                return False
                
            if self.speed_limit.get() <= 0:
                messagebox.showerror("Error", "Speed limit must be greater than zero.")
                return False
                
            if self.distance_meters.get() <= 0:
                messagebox.showerror("Error", "Distance between lines must be greater than zero.")
                return False
                
        return True

    def run_detection(self):
        """Run the traffic violation detection with the selected settings."""
        if self.processing:
            messagebox.showinfo("Info", "Detection is already running.")
            return
            
        if not self.validate_inputs():
            return
            
        # Get absolute paths
        video_path = os.path.abspath(self.video_path.get())
        output_path = os.path.abspath(self.output_path.get())
        model_path = os.path.abspath(self.model_path.get())
        
        # Prepare parameters
        params = {
            "video_path": video_path,
            "output_path": output_path,
            "model_path": model_path,
            "violation_type": self.violation_type.get(),
            "roi_data": self.roi_data
        }
        
        if self.violation_type.get() == "speed":
            params["speed_limit"] = self.speed_limit.get()
            params["distance_meters"] = self.distance_meters.get()
            
        # Add common parameters
        params["plate_reading_interval"] = self.plate_reading_interval.get()
        
        # Update the GUI
        self.processing = True
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("Running detection...")
        
        # Clear results
        if self.results_text:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Starting detection...\n\n")
            self.results_text.config(state=tk.DISABLED)
        
        # Start detection in a separate thread
        self.detector_thread = threading.Thread(target=self.detection_thread, args=(params,))
        self.detector_thread.daemon = True
        self.detector_thread.start()
        
        # Check thread status periodically
        self.root.after(100, self.check_detection_thread)

    def detection_thread(self, params):
        """Thread function to run the detection process."""
        try:
            self.update_results("Initializing detector...")
            
            # Run detection
            total_violations = run_from_gui(params)
            
            self.violation_count = total_violations
            self.update_results(f"\nDetection completed!\n"
                               f"Total violations detected: {self.violation_count}\n\n"
                               f"Processing license plates from violation images...")
            
            # Process pending violations
            processed, successful = self.violation_processor.process_pending_violations()
            
            self.update_results(f"\nLicense plate processing complete:\n"
                               f"Processed: {processed} violations\n"
                               f"Successful: {successful} plates extracted")
            
            # Refresh violations list
            self.root.after(0, self.refresh_violations_list)
            
        except Exception as e:
            self.update_results(f"Error during detection: {str(e)}")
            import traceback
            self.update_results(f"\nDetailed error:\n{traceback.format_exc()}")
    
    def update_results(self, text):
        """Update the results text widget from the detection thread."""
        self.root.after(0, self._update_results_main_thread, text)
    
    def _update_results_main_thread(self, text):
        """Update the results text widget from the main thread."""
        if self.results_text:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.insert(tk.END, f"{text}\n")
            self.results_text.see(tk.END)
            self.results_text.config(state=tk.DISABLED)

    def check_detection_thread(self):
        """Check if the detection thread has completed."""
        if self.detector_thread and not self.detector_thread.is_alive():
            self.processing = False
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set(f"Detection completed. Found {self.violation_count} violations.")
            
            # Show a message box
            messagebox.showinfo("Detection Complete", 
                              f"Detection completed!\n\n"
                              f"Total violations detected: {self.violation_count}\n"
                              f"License plates have been processed automatically.")
            
            # Switch to violations tab
            self.tab_control.select(2)  # Index 2 is the Violations tab
        elif self.processing:
            # Check again after a short delay
            self.root.after(100, self.check_detection_thread)

    def open_results_folder(self):
        """Open the folder containing detection results."""
        try:
            output_dir = config.DAILY_OUTPUT_DIR
            if os.path.exists(output_dir):
                # Use the appropriate command based on the OS
                if sys.platform == 'win32':
                    os.startfile(output_dir)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.call(['open', output_dir])
                else:  # Linux
                    import subprocess
                    subprocess.call(['xdg-open', output_dir])
            else:
                messagebox.showinfo("Info", f"Results folder not found: {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open results folder: {str(e)}")