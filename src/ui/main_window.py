import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_engine.model import AITutorModel
from utils.data_loader import DataLoader

class AIEducationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Education Tutor System - Lab Project")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f4f8')
        
        # Variables
        self.dataset = None
        self.model = None
        self.metrics = None
        self.feature_cols = []
        
        # Colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'bg': '#f0f4f8',
            'white': '#ffffff'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_frame, bg=self.colors['primary'], height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🤖 AI Education Tutor System", 
                               font=('Arial', 20, 'bold'), fg='white', bg=self.colors['primary'])
        title_label.pack(pady=10)
        
        # Create Notebook (Tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Dataset & Setup
        self.setup_tab()
        
        # Tab 2: Train Model
        self.train_tab()
        
        # Tab 3: Prediction
        self.predict_tab()
        
        # Tab 4: Dashboard
        self.dashboard_tab()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="✅ Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Dataset")
        
        # Left panel
        left_frame = tk.Frame(tab, bg=self.colors['bg'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # Upload section
        upload_frame = tk.LabelFrame(left_frame, text="📁 Load Dataset", font=('Arial', 12, 'bold'),
                                     bg=self.colors['bg'], fg=self.colors['primary'])
        upload_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(upload_frame, text="📤 Upload CSV", command=self.load_dataset,
                 bg=self.colors['secondary'], fg='white', font=('Arial', 10), padx=20, pady=10).pack(pady=10)
        
        self.file_label = tk.Label(upload_frame, text="No file loaded", bg=self.colors['bg'], fg='gray')
        self.file_label.pack(pady=5)
        
        # Dataset Info
        info_frame = tk.LabelFrame(left_frame, text="📋 Dataset Info", font=('Arial', 12, 'bold'),
                                   bg=self.colors['bg'], fg=self.colors['primary'])
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=40, font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Problem Setup
        problem_frame = tk.LabelFrame(left_frame, text="🎯 Problem Setup", font=('Arial', 12, 'bold'),
                                      bg=self.colors['bg'], fg=self.colors['primary'])
        problem_frame.pack(fill=tk.X, pady=10)
        
        problem_text = tk.Text(problem_frame, height=6, width=40, font=('Arial', 9), wrap=tk.WORD)
        problem_text.insert('1.0', """Problem: Student Performance Prediction & Learning Recommendations

Input: Student data (attendance, quiz scores, assignments, exams)
Output: Performance level & personalized study recommendations
AI Method: Machine Learning (Random Forest Classifier)""")
        problem_text.config(state=tk.DISABLED)
        problem_text.pack(padx=5, pady=5)
        
        # Right panel - Data preview
        right_frame = tk.Frame(tab, bg=self.colors['bg'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        preview_frame = tk.LabelFrame(right_frame, text="📊 Data Preview", font=('Arial', 12, 'bold'),
                                      bg=self.colors['bg'], fg=self.colors['primary'])
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(preview_frame)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🧠 Train Model")
        
        # Control panel
        control_frame = tk.Frame(tab, bg=self.colors['bg'])
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(control_frame, text="⚙️ Model Settings", font=('Arial', 14, 'bold'),
                bg=self.colors['bg'], fg=self.colors['primary']).pack(pady=10)
        
        # Model selection
        tk.Label(control_frame, text="Model:", bg=self.colors['bg'], font=('Arial', 10)).pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="Random Forest")
        model_options = ["Random Forest", "Decision Tree", "SVM", "KNN"]
        model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_var, values=model_options, width=20)
        model_dropdown.pack(pady=5)
        
        # Train button
        tk.Button(control_frame, text="🚀 Train Model", command=self.train_model,
                 bg=self.colors['success'], fg='white', font=('Arial', 12, 'bold'), padx=30, pady=15).pack(pady=20)
        
        # Results panel
        results_frame = tk.LabelFrame(control_frame, text="📈 Training Results", font=('Arial', 12, 'bold'),
                                      bg=self.colors['bg'], fg=self.colors['primary'])
        results_frame.pack(fill=tk.X, pady=10)
        
        self.metrics_text = scrolledtext.ScrolledText(results_frame, height=10, width=30, font=('Courier', 9))
        self.metrics_text.pack(padx=5, pady=5)
        
        # Right panel
        viz_frame = tk.Frame(tab, bg=self.colors['bg'])
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.train_fig = plt.Figure(figsize=(8, 6))
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=viz_frame)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def predict_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯 Predict")
        
        # Left panel - Input
        left_frame = tk.Frame(tab, bg=self.colors['bg'])
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Label(left_frame, text="👤 Student Input", font=('Arial', 14, 'bold'),
                bg=self.colors['bg'], fg=self.colors['primary']).pack(pady=10)
        
        # Input fields
        input_fields = [
            ("Attendance (%)", "attendance"),
            ("Quiz Scores (%)", "quiz_scores"),
            ("Assignment Scores (%)", "assignment_scores"),
            ("Exam Scores (%)", "exam_scores"),
            ("Study Hours/Day", "hours_studied"),
        ]
        
        self.input_vars = {}
        for label, key in input_fields:
            frame = tk.Frame(left_frame, bg=self.colors['bg'])
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=label, bg=self.colors['bg'], width=18, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=50)
            self.input_vars[key] = var
            scale = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=var, length=150)
            scale.pack(side=tk.LEFT)
            tk.Label(frame, textvariable=var, bg=self.colors['bg'], width=5).pack(side=tk.LEFT)
        
        # Predict button
        tk.Button(left_frame, text="🔮 Predict", command=self.predict_student,
                 bg=self.colors['secondary'], fg='white', font=('Arial', 12, 'bold'), padx=20, pady=10).pack(pady=20)
        
        # Result panel
        result_frame = tk.LabelFrame(left_frame, text="📋 Result", font=('Arial', 12, 'bold'),
                                     bg=self.colors['bg'], fg=self.colors['primary'])
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, width=35, font=('Arial', 10))
        self.result_text.pack(padx=5, pady=5)
        
        # Right panel
        right_frame = tk.Frame(tab, bg=self.colors['bg'])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.pred_fig = plt.Figure(figsize=(8, 6))
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=right_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def dashboard_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Dashboard")
        
        # Stats cards
        stats_frame = tk.Frame(tab, bg=self.colors['bg'])
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        stats_data = [
            ("📚 Total Students", "0", self.colors['secondary']),
            ("🎯 Average Score", "0%", self.colors['success']),
            ("📈 Best Performance", "0%", self.colors['warning']),
            ("🏆 Total Quizzes", "0", self.colors['primary'])
        ]
        
        self.stats_labels = []
        for i, (label, value, color) in enumerate(stats_data):
            frame = tk.Frame(stats_frame, bg=color, relief=tk.RAISED, bd=2)
            frame.grid(row=0, column=i, padx=10, pady=5, sticky="nsew")
            tk.Label(frame, text=label, font=('Arial', 10), bg=color, fg='white').pack(pady=(10,0))
            stat_label = tk.Label(frame, text=value, font=('Arial', 24, 'bold'), bg=color, fg='white')
            stat_label.pack(pady=10)
            self.stats_labels.append(stat_label)
        
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_columnconfigure(2, weight=1)
        stats_frame.grid_columnconfigure(3, weight=1)
        
        # Charts
        chart_frame = tk.Frame(tab, bg=self.colors['bg'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dash_fig = plt.Figure(figsize=(10, 6))
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=chart_frame)
        self.dash_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    # ============== Core Functions ==============
    
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if file_path:
            try:
                self.dataset = DataLoader.load_data(file_path)
                self.file_label.config(text=f"✅ Loaded: {os.path.basename(file_path)}", fg='green')
                
                # Update info
                info = DataLoader.get_dataset_info(self.dataset)
                self.info_text.delete('1.0', tk.END)
                self.info_text.insert('1.0', info)
                
                # Update tree
                self.update_treeview()
                
                # Update stats
                self.update_stats()
                
                self.status_bar.config(text=f"✅ Dataset loaded: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Dataset loaded successfully!\nRows: {len(self.dataset)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                self.status_bar.config(text=f"❌ Error: {str(e)}")
    
    def update_treeview(self):
        if self.dataset is None or self.dataset.empty:
            return
        
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Set columns
        columns = list(self.dataset.columns)
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add data
        for _, row in self.dataset.head(20).iterrows():
            self.tree.insert('', 'end', values=list(row))
    
    def train_model(self):
        if self.dataset is None or self.dataset.empty:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
        
        try:
            self.status_bar.config(text="⏳ Training model...")
            self.root.update()
            
            # Train model
            model_type = self.model_var.get()
            self.model, self.metrics, _, _ = AITutorModel.train(self.dataset, model_type=model_type)
            
            # Display metrics
            self.metrics_text.delete('1.0', tk.END)
            metrics_str = AITutorModel.display_metrics(self.metrics)
            self.metrics_text.insert('1.0', metrics_str)
            
            # Update dashboard
            self.update_dashboard()
            
            # Show confusion matrix
            self.show_confusion_matrix()
            
            self.status_bar.config(text="✅ Model trained successfully!")
            messagebox.showinfo("Success", f"Model trained!\nAccuracy: {self.metrics['accuracy']:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_bar.config(text=f"❌ Error: {str(e)}")
    
    def predict_student(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
        
        try:
            # Get input
            student_data = {
                'attendance': self.input_vars['attendance'].get(),
                'quiz_scores': self.input_vars['quiz_scores'].get(),
                'assignment_scores': self.input_vars['assignment_scores'].get(),
                'exam_scores': self.input_vars['exam_scores'].get(),
                'hours_studied': self.input_vars['hours_studied'].get(),
            }
            
            # Predict
            prediction, recommendation, probabilities = AITutorModel.predict(self.model, student_data)
            
            # Display result
            self.result_text.delete('1.0', tk.END)
            result_str = AITutorModel.display_prediction(student_data, prediction, recommendation, probabilities)
            self.result_text.insert('1.0', result_str)
            
            # Show visualization
            self.show_prediction_viz(student_data, prediction, probabilities)
            
            self.status_bar.config(text=f"✅ Prediction: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_bar.config(text=f"❌ Error: {str(e)}")
    
    def update_stats(self):
        if self.dataset is not None and not self.dataset.empty:
            stats = DataLoader.get_stats(self.dataset)
            self.stats_labels[0].config(text=str(stats['total_students']))
            self.stats_labels[1].config(text=f"{stats['avg_score']:.1f}%")
            self.stats_labels[2].config(text=f"{stats['best_score']:.1f}%")
            self.stats_labels[3].config(text=str(stats['total_quizzes']))
    
    def update_dashboard(self):
        if self.dataset is not None and not self.dataset.empty:
            self.dash_fig.clear()
            ax1 = self.dash_fig.add_subplot(121)
            ax2 = self.dash_fig.add_subplot(122)
            
            if 'quiz_scores' in self.dataset.columns:
                ax1.hist(self.dataset['quiz_scores'], bins=20, color='skyblue', edgecolor='black')
                ax1.set_title('Quiz Scores Distribution')
                ax1.set_xlabel('Score')
                ax1.set_ylabel('Frequency')
            
            if 'performance_level' in self.dataset.columns:
                levels = self.dataset['performance_level'].value_counts()
                ax2.pie(levels.values, labels=levels.index, autopct='%1.1f%%')
                ax2.set_title('Performance Levels')
            
            self.dash_fig.tight_layout()
            self.dash_canvas.draw()
    
    def show_confusion_matrix(self):
        if self.metrics and 'confusion_matrix' in self.metrics:
            self.train_fig.clear()
            ax = self.train_fig.add_subplot(111)
            
            cm = self.metrics['confusion_matrix']
            im = ax.imshow(cm, cmap='Blues')
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center')
            
            self.train_fig.colorbar(im)
            self.train_canvas.draw()
    
    def show_prediction_viz(self, student_data, prediction, probabilities):
        self.pred_fig.clear()
        
        ax1 = self.pred_fig.add_subplot(121)
        features = list(student_data.keys())
        values = list(student_data.values())
        colors = ['#3498db' if v < 50 else '#27ae60' for v in values]
        ax1.bar(features, values, color=colors)
        ax1.set_title('Student Input Data')
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Score (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = self.pred_fig.add_subplot(122)
        levels = list(probabilities.keys())
        probs = list(probabilities.values())
        colors2 = ['#e74c3c' if l == prediction else '#3498db' for l in levels]
        ax2.bar(levels, probs, color=colors2)
        ax2.set_title(f'Prediction: {prediction}')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        
        self.pred_fig.tight_layout()
        self.pred_canvas.draw()