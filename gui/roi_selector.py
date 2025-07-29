"""
ROI selection interface for traffic violation detector.
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class ROISelector:
    """
    Class to handle selection of ROI points on the video frame.
    """
    def __init__(self, master, frame, roi_type="redlight"):
        self.master = master
        self.frame = frame.copy()
        self.height, self.width = self.frame.shape[:2]
        self.roi_type = roi_type
        
        # For red light, we need a polygon
        # For speed, we need two lines (entry and exit)
        if roi_type == "redlight":
            self.points = []
            self.max_points = 4
            self.roi_text = "Select 4 points for the Red Light ROI"
        else:  # speed
            self.entry_line = []
            self.exit_line = []
            self.current_line = "entry"
            self.max_points = 2
            self.roi_text = "Select 2 points for Entry Line (FIRST line vehicles will cross)"
        
        # Create a new window
        self.roi_window = tk.Toplevel(master)
        self.roi_window.title("ROI Selection")
        self.roi_window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Set minimum window size
        self.roi_window.minsize(800, 600)
        
        # Create a main frame
        main_frame = ttk.Frame(self.roi_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a canvas to display the frame
        self.canvas_width = min(1280, self.width)
        self.canvas_height = int(self.height * (self.canvas_width / self.width))
        self.scale_factor = self.canvas_width / self.width
        
        # Create the widgets
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, 
                               bd=2, relief=tk.SUNKEN)
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Instructions label
        self.instruction_label = ttk.Label(main_frame, text=self.roi_text, font=("Arial", 12))
        self.instruction_label.pack(pady=5)
        
        # Status label to show clicked points
        self.status_label = ttk.Label(main_frame, text="No points selected")
        self.status_label.pack(pady=5)
        
        # Keyboard instructions
        keyboard_text = "Press ENTER or 'S' to save ROI after selecting all points. Press 'C' to clear points."
        self.keyboard_label = ttk.Label(main_frame, text=keyboard_text, font=("Arial", 10, "italic"))
        self.keyboard_label.pack(pady=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10, fill=tk.X)
        
        # Create buttons with ttk for better appearance
        self.clear_btn = ttk.Button(btn_frame, text="Clear Points (C)", command=self.clear_points)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Make save button more prominent
        self.save_btn = ttk.Button(btn_frame, text="Save ROI (Enter/S)", command=self.save_roi, 
                                  state=tk.DISABLED, style='Accent.TButton')
        self.save_btn.pack(side=tk.RIGHT, padx=10)
        
        # Create a custom style for the accent button
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        
        # Display the frame on the canvas
        self.display_frame()
        
        # Bind mouse click event to canvas
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Bind keyboard events to window
        self.roi_window.bind("<Return>", lambda event: self.try_save_roi())  # Enter key
        self.roi_window.bind("s", lambda event: self.try_save_roi())         # 's' key
        self.roi_window.bind("S", lambda event: self.try_save_roi())         # 'S' key (capital)
        self.roi_window.bind("c", lambda event: self.clear_points())         # 'c' key
        self.roi_window.bind("C", lambda event: self.clear_points())         # 'C' key (capital)
        
        # Center the window
        self.center_window()
        
        # Give the window focus
        self.roi_window.focus_force()
        
        # Print debug info
        print(f"ROI window created. Press Enter or 'S' to save after selecting all points.")
    
    def center_window(self):
        """Center the ROI window on the screen."""
        self.roi_window.update_idletasks()
        width = self.roi_window.winfo_width()
        height = self.roi_window.winfo_height()
        x = (self.roi_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.roi_window.winfo_screenheight() // 2) - (height // 2)
        self.roi_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def display_frame(self):
        """Display the frame on the canvas with current ROI points."""
        # Resize the frame for display
        display_frame = cv2.resize(self.frame.copy(), (self.canvas_width, self.canvas_height))
        
        # Draw existing points
        if self.roi_type == "redlight":
            # Draw red light ROI polygon
            for i, point in enumerate(self.points):
                # Scale point coordinates
                scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                cv2.circle(display_frame, scaled_point, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (scaled_point[0] + 5, scaled_point[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            # Draw polygon if we have at least 2 points
            if len(self.points) >= 2:
                scaled_points = [(int(p[0] * self.scale_factor), int(p[1] * self.scale_factor)) for p in self.points]
                pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                
            # Update status label
            self.status_label.config(text=f"Selected points: {len(self.points)}/{self.max_points}")
            
            # Update save button state
            can_save = len(self.points) == self.max_points
            self.save_btn.config(state=tk.NORMAL if can_save else tk.DISABLED)
            
        else:
            # Draw speed lines
            # Entry line (blue)
            for i, point in enumerate(self.entry_line):
                scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                cv2.circle(display_frame, scaled_point, 5, (255, 0, 0), -1)
                cv2.putText(display_frame, f"E{i+1}", (scaled_point[0] + 5, scaled_point[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Exit line (red)
            for i, point in enumerate(self.exit_line):
                scaled_point = (int(point[0] * self.scale_factor), int(point[1] * self.scale_factor))
                cv2.circle(display_frame, scaled_point, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"X{i+1}", (scaled_point[0] + 5, scaled_point[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw lines if we have 2 points
            if len(self.entry_line) == 2:
                p1 = (int(self.entry_line[0][0] * self.scale_factor), int(self.entry_line[0][1] * self.scale_factor))
                p2 = (int(self.entry_line[1][0] * self.scale_factor), int(self.entry_line[1][1] * self.scale_factor))
                cv2.line(display_frame, p1, p2, (255, 0, 0), 2)
                cv2.putText(display_frame, "ENTRY LINE", 
                            (p1[0] + 20, p1[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            if len(self.exit_line) == 2:
                p1 = (int(self.exit_line[0][0] * self.scale_factor), int(self.exit_line[0][1] * self.scale_factor))
                p2 = (int(self.exit_line[1][0] * self.scale_factor), int(self.exit_line[1][1] * self.scale_factor))
                cv2.line(display_frame, p1, p2, (0, 0, 255), 2)
                cv2.putText(display_frame, "EXIT LINE", 
                            (p1[0] + 20, p1[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Update status label
            if self.current_line == "entry":
                self.status_label.config(text=f"Entry line points: {len(self.entry_line)}/{self.max_points}")
            else:
                self.status_label.config(text=f"Exit line points: {len(self.exit_line)}/{self.max_points}")
                
            # Update save button state
            can_save = len(self.entry_line) == self.max_points and len(self.exit_line) == self.max_points
            self.save_btn.config(state=tk.NORMAL if can_save else tk.DISABLED)
        
        # Convert the frame to a format that Tkinter can use
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the canvas
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    
    def on_click(self, event):
        """Handle mouse click events on the canvas."""
        # Convert canvas coordinates to original frame coordinates
        x = int(event.x / self.scale_factor)
        y = int(event.y / self.scale_factor)
        
        if self.roi_type == "redlight":
            # Add point to red light ROI
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                print(f"Added point {len(self.points)}: ({x}, {y})")
        else:
            # Add point to speed detector lines
            if self.current_line == "entry":
                if len(self.entry_line) < self.max_points:
                    self.entry_line.append((x, y))
                    print(f"Added entry point {len(self.entry_line)}: ({x}, {y})")
                    
                    if len(self.entry_line) == self.max_points:
                        self.current_line = "exit"
                        self.instruction_label.config(text="Select 2 points for Exit Line (SECOND line vehicles will cross)")
                        self.instruction_label.config(foreground="red")  # Make the text red for emphasis
            else:  # exit line
                if len(self.exit_line) < self.max_points:
                    self.exit_line.append((x, y))
                    print(f"Added exit point {len(self.exit_line)}: ({x}, {y})")
        
        # Update the display
        self.display_frame()
        
        # Check if all points are selected and we can save
        if self.can_save_roi():
            self.keyboard_label.config(text="All points selected! Press ENTER or 'S' to save ROI.", foreground="green")
    
    def can_save_roi(self):
        """Check if we have all the points needed to save the ROI."""
        if self.roi_type == "redlight":
            return len(self.points) == self.max_points
        else:
            return len(self.entry_line) == self.max_points and len(self.exit_line) == self.max_points
    
    def try_save_roi(self):
        """Try to save the ROI if all points are selected."""
        if self.can_save_roi():
            self.save_roi()
        else:
            # Show a message if we don't have all points yet
            points_needed = ""
            if self.roi_type == "redlight":
                points_needed = f"{self.max_points - len(self.points)} more points"
            else:
                if len(self.entry_line) < self.max_points:
                    points_needed = f"{self.max_points - len(self.entry_line)} more entry line points"
                else:
                    points_needed = f"{self.max_points - len(self.exit_line)} more exit line points"
                
            self.status_label.config(text=f"Cannot save yet. Need {points_needed}.", foreground="red")
            self.roi_window.after(2000, lambda: self.status_label.config(foreground="black"))
    
    def clear_points(self):
        """Clear all selected points."""
        if self.roi_type == "redlight":
            self.points = []
            print("Cleared red light ROI points")
        else:
            if self.current_line == "exit" and len(self.exit_line) > 0:
                self.exit_line = []
                print("Cleared exit line points")
            else:
                self.entry_line = []
                self.current_line = "entry"
                self.instruction_label.config(text="Select 2 points for Entry Line (FIRST line vehicles will cross)")
                self.instruction_label.config(foreground="black")  # Reset text color
                print("Cleared entry line points")
        
        # Reset keyboard label
        self.keyboard_label.config(text="Press ENTER or 'S' to save ROI after selecting all points. Press 'C' to clear points.",
                                foreground="black")
        
        # Update the display
        self.display_frame()
    
    def save_roi(self):
        """Save the selected ROI points and close the window."""
        if self.roi_type == "redlight":
            result = {"type": "redlight", "points": self.points}
            print(f"Saving red light ROI: {self.points}")
        else:
            result = {"type": "speed", "entry_line": self.entry_line, "exit_line": self.exit_line}
            print(f"Saving speed ROI: Entry={self.entry_line}, Exit={self.exit_line}")
        
        # Get the GUI app instance from the root window
        if hasattr(self.master, 'gui_app'):
            self.master.gui_app.set_roi_result(result)
            print("ROI saved to GUI app")
        else:
            print("Warning: Could not find gui_app attribute in master")
        
        self.roi_window.destroy()
    
    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Cancel ROI Selection", "Are you sure you want to cancel ROI selection?"):
            # Get the GUI app instance from the root window
            if hasattr(self.master, 'gui_app'):
                self.master.gui_app.set_roi_result(None)
                print("ROI selection cancelled")
            
            self.roi_window.destroy()