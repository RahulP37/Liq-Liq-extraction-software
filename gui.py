try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input, TimeDistributed, Lambda
    from tensorflow.keras.callbacks import EarlyStopping, Callback
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

import sys
import os
import numpy as np
import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QGroupBox, QLabel, 
                             QDoubleSpinBox, QSpinBox, QRadioButton, QComboBox, 
                             QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox, QScrollArea, QStackedWidget, QLineEdit)
from PyQt5.QtCore import Qt

# --- Custom Keras Callback to update the GUI during training ---
class GUIUpdateCallback(Callback):
    def __init__(self, status_label, model_name):
        super().__init__()
        self.status_label = status_label
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # Update the GUI every 10 epochs so it doesn't freeze
        if epoch % 10 == 0:
            mae = logs.get('mae', 0)
            val_mae = logs.get('val_mae', 0)
            self.status_label.setText(f"Status: Training {self.model_name} | Epoch {epoch} | Val MAE: {val_mae:.2f}")
            QApplication.processEvents()

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.is_3d = is_3d
        if is_3d: self.axes = self.fig.add_subplot(111, projection='3d')
        else: self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.mpl_connect("scroll_event", self.zoom_with_mouse_wheel)

    def zoom_with_mouse_wheel(self, event):
        if event.inaxes != self.axes or self.is_3d: return
        base_scale = 1.1
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale if event.button == 'down' else 1
        cur_xlim, cur_ylim = self.axes.get_xlim(), self.axes.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_width, new_height = (cur_xlim[1] - cur_xlim[0]) * scale_factor, (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx, rely = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]), (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.axes.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
        self.axes.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
        self.draw()

class ExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("End-to-End Digital Twin & AI Surrogate Model")
        self.setGeometry(100, 100, 1150, 800) 

        # Directories
        self.model_dir = "extraction_models" 
        self.use_pretrained = True
        self.custom_data_dir = "custom_models"

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_setup_screen()
        self.init_main_dashboard()
        
        self.stacked_widget.addWidget(self.setup_widget)
        self.stacked_widget.addWidget(self.main_widget)
        
        # Load default Group 2 Data
        self.load_eqm_data(
            "0.0155, 0.0170, 0.0250, 0.0380, 0.0600, 0.1220, 0.2250",
            "0.0000, 0.0285, 0.1170, 0.2050, 0.2620, 0.3280, 0.3460",
            "0.9788, 0.9533, 0.8570, 0.7350, 0.6090, 0.4720, 0.3540",
            "0.0000, 0.0187, 0.0890, 0.1730, 0.2460, 0.3080, 0.3360"
        )

    # ==========================================
    # SCREEN 1: SETUP WIZARD (Data & Training)
    # ==========================================
    def init_setup_screen(self):
        self.setup_widget = QWidget()
        layout = QVBoxLayout(self.setup_widget)
        
        title = QLabel("Digital Twin Setup Wizard")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        mode_box = QGroupBox("1. AI Model Selection")
        mode_box.setStyleSheet("font-weight: bold;")
        mode_layout = QVBoxLayout()
        self.radio_pretrained = QRadioButton("Use Pretrained Models (Group 2 Defaults in 'extraction_models')")
        self.radio_custom = QRadioButton("Train Custom Models (Provide new Equilibrium Data)")
        self.radio_pretrained.setChecked(True)
        self.radio_pretrained.toggled.connect(self.toggle_setup_mode)
        mode_layout.addWidget(self.radio_pretrained)
        mode_layout.addWidget(self.radio_custom)
        mode_box.setLayout(mode_layout)
        layout.addWidget(mode_box)

        self.eqm_box = QGroupBox("2. Custom Equilibrium Data (Comma Separated)")
        self.eqm_box.setStyleSheet("font-weight: bold;")
        eqm_layout = QFormLayout()
        self.in_xb = QLineEdit("0.0155, 0.0170, 0.0250, 0.0380, 0.0600, 0.1220, 0.2250")
        self.in_xc = QLineEdit("0.0000, 0.0285, 0.1170, 0.2050, 0.2620, 0.3280, 0.3460")
        self.in_yb = QLineEdit("0.9788, 0.9533, 0.8570, 0.7350, 0.6090, 0.4720, 0.3540")
        self.in_yc = QLineEdit("0.0000, 0.0187, 0.0890, 0.1730, 0.2460, 0.3080, 0.3360")
        
        # Reset font weight for inputs
        for w in [self.in_xb, self.in_xc, self.in_yb, self.in_yc]:
            w.setStyleSheet("font-weight: normal;")
            
        eqm_layout.addRow("Raffinate Solvent (xB):", self.in_xb)
        eqm_layout.addRow("Raffinate Solute (xC):", self.in_xc)
        eqm_layout.addRow("Extract Solvent (yB):", self.in_yb)
        eqm_layout.addRow("Extract Solute (yC):", self.in_yc)
        self.eqm_box.setLayout(eqm_layout)
        self.eqm_box.setEnabled(False)
        layout.addWidget(self.eqm_box)

        self.action_box = QGroupBox("3. Custom Model Generation Pipeline")
        self.action_box.setStyleSheet("font-weight: bold;")
        action_layout = QVBoxLayout()
        
        self.btn_generate = QPushButton("Step 1: Generate Synthetic Dataset (Math Simulator)")
        self.btn_generate.setStyleSheet("background-color: #3498db; color: white; padding: 8px;")
        self.btn_generate.clicked.connect(self.generate_custom_dataset)
        
        self.btn_train = QPushButton("Step 2: Train AI Surrogate Models")
        self.btn_train.setStyleSheet("background-color: #9b59b6; color: white; padding: 8px;")
        self.btn_train.clicked.connect(self.train_custom_models)
        self.btn_train.setEnabled(False)
        
        self.status_lbl = QLabel("Status: Ready to launch with Pretrained models.")
        self.status_lbl.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px; margin-top: 10px;")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        
        action_layout.addWidget(self.btn_generate)
        action_layout.addWidget(self.btn_train)
        action_layout.addWidget(self.status_lbl)
        self.action_box.setLayout(action_layout)
        self.action_box.setEnabled(False)
        layout.addWidget(self.action_box)

        layout.addStretch()

        self.btn_launch = QPushButton("LAUNCH DIGITAL TWIN DASHBOARD")
        self.btn_launch.setStyleSheet("background-color: #2ecc71; color: white; font-size: 18px; padding: 20px; font-weight: bold; border-radius: 5px;")
        self.btn_launch.clicked.connect(self.launch_dashboard)
        layout.addWidget(self.btn_launch)

    def toggle_setup_mode(self):
        self.use_pretrained = self.radio_pretrained.isChecked()
        self.eqm_box.setEnabled(not self.use_pretrained)
        self.action_box.setEnabled(not self.use_pretrained)
        if self.use_pretrained:
            self.model_dir = "extraction_models"
            self.status_lbl.setText("Status: Ready to launch with Pretrained models.")
        else:
            self.model_dir = self.custom_data_dir
            self.status_lbl.setText("Status: Waiting to generate dataset...")

    def load_eqm_data(self, xb_str, xc_str, yb_str, yc_str):
        try:
            self.xB = np.array([float(x.strip()) for x in xb_str.split(',')])
            self.xC = np.array([float(x.strip()) for x in xc_str.split(',')])
            self.yB = np.array([float(x.strip()) for x in yb_str.split(',')])
            self.yC = np.array([float(x.strip()) for x in yc_str.split(',')])
            self.eq_interp = interp1d(self.xC, self.yC, kind='cubic', fill_value="extrapolate")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Data Error", f"Failed to parse equilibrium data:\n{e}")
            return False

    # --- DATASET GENERATOR ---
    def generate_custom_dataset(self):
        if not self.load_eqm_data(self.in_xb.text(), self.in_xc.text(), self.in_yb.text(), self.in_yc.text()): return
        
        self.status_lbl.setText("Status: Generating Data via mathematical solver... Please wait.")
        QApplication.processEvents()

        os.makedirs(self.custom_data_dir, exist_ok=True)
        
        # Grid parameters from your provided code
        F_values = [800 ]
        S_values = [1000, 1300 ]
        xcf_values = [0.25, 0.35]
        N_stages = 20

        cross_rows, count_rows = [], []
        
        total_runs = len(F_values) * len(S_values) * len(xcf_values)
        current_run = 0

        for F in F_values:
            for S in S_values:
                for xcf in xcf_values:
                    current_run += 1
                    self.status_lbl.setText(f"Status: Solving physical equations... {current_run}/{total_runs}")
                    QApplication.processEvents()
                    
                    # Run Rigorous Math Solver
                    x_cr, _ = self.solve_crosscurrent(F, S, xcf, N_stages)
                    x_co, _ = self.solve_countercurrent(F, S, xcf, N_stages)
                    
                    # Calculate % Removal and format exactly like the provided training script
                    row_cr = [F, S, xcf] + [((xcf - x_cr[i]) / xcf) * 100 for i in range(N_stages)]
                    row_co = [F, S, xcf] + [((xcf - x_co[i]) / xcf) * 100 for i in range(N_stages)]
                    
                    cross_rows.append(row_cr)
                    count_rows.append(row_co)

        cols = ['Feed Flowrate', 'Solvent Flowrate', 'xcf'] + [f'Stage_{i}' for i in range(1, 21)]
        
        # Mode='w' ensures previous custom data is overwritten!
        pd.DataFrame(cross_rows, columns=cols).to_csv(f"{self.custom_data_dir}/cross_data.csv", index=False, mode='w')
        pd.DataFrame(count_rows, columns=cols).to_csv(f"{self.custom_data_dir}/count_data.csv", index=False, mode='w')

        self.status_lbl.setText("Status: Synthetic Dataset Generated Successfully! Ready to Train.")
        self.btn_train.setEnabled(True)

    # --- AI TRAINER ---
    def train_custom_models(self):
        if not AI_AVAILABLE:
            QMessageBox.critical(self, "Error", "TensorFlow is not installed!")
            return
            
        try:
            df_cross = pd.read_csv(f"{self.custom_data_dir}/cross_data.csv").sample(frac=1, random_state=42).reset_index(drop=True)
            df_counter = pd.read_csv(f"{self.custom_data_dir}/count_data.csv").sample(frac=1, random_state=42).reset_index(drop=True)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "Dataset not found. Generate it first.")
            return

        def build_and_train(df, mode_str):
            num_stages = 20
            N_samples = len(df)
            
            # --- Parse Data (Just like your provided training script) ---
            X_rnn = np.zeros((N_samples, num_stages, 4))
            y_rnn = np.zeros((N_samples, num_stages, 1))

            for i in range(N_samples):
                row = df.iloc[i]
                feed, solvent, xcf = row['Feed Flowrate'], row['Solvent Flowrate'], row['xcf']
                for stage in range(1, num_stages + 1):
                    X_rnn[i, stage-1] = [feed, solvent, xcf, stage]
                    y_rnn[i, stage-1, 0] = row[f'Stage_{stage}']

            X_ann = X_rnn.reshape(-1, 4)
            y_ann = y_rnn.reshape(-1, 1)

            scaler = StandardScaler().fit(X_ann)
            joblib.dump(scaler, f"{self.custom_data_dir}/scaler_{mode_str}.pkl")

            X_ann_scaled = scaler.transform(X_ann)
            X_rnn_scaled = X_ann_scaled.reshape(-1, num_stages, 4)

            # Splits
            X_a_train, _, y_a_train, _ = train_test_split(X_ann_scaled, y_ann, test_size=0.2, random_state=42)
            X_r_train, _, y_r_train, _ = train_test_split(X_rnn_scaled, y_rnn, test_size=0.2, random_state=42)

            early_stop = EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)

            # --- Define your Architectures ---
            models = {
                'ann': Sequential([
                    Input(shape=(4,)),
                    Dense(64, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(1, activation='sigmoid'),
                    Lambda(lambda x: x * 100)
                ]),
                'rnn': Sequential([
                    Input(shape=(20, 4)),
                    SimpleRNN(64, return_sequences=True),
                    TimeDistributed(Dense(1, activation='sigmoid')),
                    TimeDistributed(Lambda(lambda x: x * 100))
                ]),
                'lstm': Sequential([
                    Input(shape=(20, 4)),
                    LSTM(64, return_sequences=True),
                    TimeDistributed(Dense(1, activation='sigmoid')),
                    TimeDistributed(Lambda(lambda x: x * 100))
                ]),
                'gru': Sequential([
                    Input(shape=(20, 4)),
                    GRU(64, return_sequences=True),
                    TimeDistributed(Dense(1, activation='sigmoid')),
                    TimeDistributed(Lambda(lambda x: x * 100))
                ])
            }

            for name, model in models.items():
                self.status_lbl.setText(f"Status: Starting {name.upper()} ({mode_str})...")
                QApplication.processEvents()
                
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                gui_callback = GUIUpdateCallback(self.status_lbl, f"{name.upper()} ({mode_str})")
                
                epochs = 10 if name == 'ann' else 20
                
                if name == 'ann':
                    model.fit(X_a_train, y_a_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[gui_callback], verbose=0)
                else:
                    model.fit(X_r_train, y_r_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stop, gui_callback], verbose=0)
                
                model.save(f"{self.custom_data_dir}/{name}_{mode_str}.keras")

        # Train both process types
        build_and_train(df_cross, "crosscurrent")
        build_and_train(df_counter, "countercurrent")
        
        self.status_lbl.setText("Status: All 8 Neural Networks Trained & Saved! Ready to Launch.")
        self.status_lbl.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")

    def launch_dashboard(self):
        if not self.use_pretrained:
            if not self.load_eqm_data(self.in_xb.text(), self.in_xc.text(), self.in_yb.text(), self.in_yc.text()): return
        
        self.plot_equilibrium() 
        self.stacked_widget.setCurrentIndex(1) 

    # ==========================================
    # SCREEN 2: MAIN DASHBOARD 
    # ==========================================
    def init_main_dashboard(self):
        self.main_widget = QWidget()
        main_layout = QHBoxLayout(self.main_widget)

        # --- LEFT PANEL (Inputs) ---
        left_panel = QVBoxLayout()
        param_box = QGroupBox("1. Process Parameters")
        param_layout = QFormLayout()
        
        self.input_F = QDoubleSpinBox(); self.input_F.setRange(100, 5000); self.input_F.setValue(1000.0)
        self.input_S = QDoubleSpinBox(); self.input_S.setRange(100, 5000); self.input_S.setValue(1300.0)
        self.input_xcf = QDoubleSpinBox(); self.input_xcf.setRange(0.01, 1.0); self.input_xcf.setValue(0.35); self.input_xcf.setSingleStep(0.01)
        self.input_N = QSpinBox(); self.input_N.setRange(1, 20); self.input_N.setValue(3)
        
        param_layout.addRow("Feed (F):", self.input_F)
        param_layout.addRow("Solvent (S):", self.input_S)
        param_layout.addRow("Feed Comp (xcf):", self.input_xcf)
        param_layout.addRow("Stages (N):", self.input_N)
        param_box.setLayout(param_layout)
        left_panel.addWidget(param_box)

        op_box = QGroupBox("2. Operation Type")
        op_layout = QVBoxLayout()
        self.radio_cross = QRadioButton("Cross-current"); self.radio_cross.setChecked(True)
        self.radio_counter = QRadioButton("Counter-current")
        op_layout.addWidget(self.radio_cross)
        op_layout.addWidget(self.radio_counter)
        op_box.setLayout(op_layout)
        left_panel.addWidget(op_box)

        method_box = QGroupBox("3. Solver Method")
        method_layout = QVBoxLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Mathematical Solver (fsolve)", "AI: ANN", "AI: SimpleRNN", "AI: LSTM", "AI: GRU"])
        method_layout.addWidget(self.combo_method)
        method_box.setLayout(method_layout)
        left_panel.addWidget(method_box)

        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 15px; font-size: 14px;")
        self.btn_run.clicked.connect(self.run_calculations) 
        left_panel.addWidget(self.btn_run)
        
        left_panel.addStretch()
        
        # Back Button to reset
        btn_back = QPushButton("← Configure New Data")
        btn_back.setStyleSheet("background-color: #7f8c8d; color: white; padding: 10px; font-weight: bold;")
        btn_back.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        left_panel.addWidget(btn_back)

        # --- RIGHT PANEL (Tabs) ---
        self.tabs = QTabWidget()
        
        self.tab_dashboard = QWidget(); dash_layout = QVBoxLayout(self.tab_dashboard)
        self.lbl_result = QLabel("Enter parameters and click 'RUN SIMULATION'")
        self.lbl_result.setStyleSheet("font-size: 18px; color: #d32f2f; font-weight: bold;")
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["Stage", "Raffinate Solute (x_out)", "Extract Solute (y_out)", "Total Removal (%)"])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        dash_layout.addWidget(self.lbl_result); dash_layout.addWidget(self.table_results)
        self.tabs.addTab(self.tab_dashboard, "1. Dashboard")

        self.tab_comparison = QWidget(); comp_layout = QVBoxLayout(self.tab_comparison)
        self.canvas_comp = PlotCanvas(self); self.toolbar_comp = NavigationToolbar(self.canvas_comp, self)
        comp_layout.addWidget(self.toolbar_comp); comp_layout.addWidget(self.canvas_comp)
        self.tabs.addTab(self.tab_comparison, "2. Comparison")

        self.tab_heatmap = QWidget(); heat_layout = QVBoxLayout(self.tab_heatmap)
        self.canvas_heat = PlotCanvas(self); self.toolbar_heat = NavigationToolbar(self.canvas_heat, self)
        heat_layout.addWidget(self.toolbar_heat); heat_layout.addWidget(self.canvas_heat)
        self.tabs.addTab(self.tab_heatmap, "3. Heatmaps")

        self.tab_3d = QWidget(); layout_3d = QVBoxLayout(self.tab_3d)
        self.canvas_3d = PlotCanvas(self, is_3d=True); self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
        layout_3d.addWidget(self.toolbar_3d); layout_3d.addWidget(self.canvas_3d)
        self.tabs.addTab(self.tab_3d, "4. 3D Surface Plots")

        self.tab_equilibrium = QWidget(); eq_layout = QVBoxLayout(self.tab_equilibrium)
        self.canvas_eq = PlotCanvas(self); self.toolbar_eq = NavigationToolbar(self.canvas_eq, self)
        eq_layout.addWidget(self.toolbar_eq); eq_layout.addWidget(self.canvas_eq)
        self.tabs.addTab(self.tab_equilibrium, "5. Equilibrium Curve")

        self.tab_contour = QWidget(); contour_outer_layout = QVBoxLayout(self.tab_contour)
        self.scroll_contour = QScrollArea(); self.scroll_contour.setWidgetResizable(True)
        self.contour_container = QWidget(); self.contour_inner_layout = QVBoxLayout(self.contour_container)
        self.canvas_contour = PlotCanvas(self, width=10, height=40); self.canvas_contour.setMinimumHeight(4500) 
        self.toolbar_contour = NavigationToolbar(self.canvas_contour, self)
        self.contour_inner_layout.addWidget(self.canvas_contour); self.scroll_contour.setWidget(self.contour_container)
        contour_outer_layout.addWidget(self.toolbar_contour); contour_outer_layout.addWidget(self.scroll_contour)
        self.tabs.addTab(self.tab_contour, "6. Model Contour Plots")

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.tabs, 3)

    # ==========================================
    # MATH ENGINE
    # ==========================================
    def equilibrium(self, x):
        return np.clip(self.eq_interp(x), 0, 1)

    def solve_crosscurrent(self, F, S, xcf, N):
        x_stages, y_stages = [], []
        xin, Rin = xcf, F
        def cross_stage(vars, R_in, x_in, Solv):
            xout, Rout, Eout = vars
            yout = self.equilibrium(xout)
            return [R_in + Solv - Rout - Eout, R_in*x_in - Rout*xout - Eout*yout, yout - self.equilibrium(xout)]

        for i in range(N):
            sol = fsolve(cross_stage, [xin*0.7, Rin*0.9, S*0.5], args=(Rin, xin, S))
            xout, Rout, Eout = sol
            yout = self.equilibrium(xout)
            x_stages.append(xout)
            y_stages.append(yout)
            xin, Rin = xout, Rout
        return x_stages, y_stages

    def solve_countercurrent(self, F, S, xcf, N):
        def obj(vars):
            x, y = vars[:N], vars[N:]
            eqs = []
            for i in range(N):
                x_in = xcf if i == 0 else x[i-1]
                y_in = 0 if i == N-1 else y[i+1]
                eqs.append(F*x_in + S*y_in - F*x[i] - S*y[i])
                eqs.append(y[i] - self.equilibrium(x[i]))
            return eqs
        sol = fsolve(obj, np.concatenate([np.linspace(xcf*0.8, xcf*0.1, N), self.equilibrium(np.linspace(xcf*0.8, xcf*0.1, N))]))
        return sol[:N].tolist(), sol[N:].tolist()

    # ==========================================
    # AI ENGINE
    # ==========================================
    def solve_ai_model(self, F, S, xcf, N, is_cross, model_type):
        mode_str = "crosscurrent" if is_cross else "countercurrent"
        model_name = model_type.split(": ")[1].lower().replace("simplernn", "rnn")
        
        # Uses DYNAMIC model_dir based on user setup selection!
        model_path = f"{self.model_dir}/{model_name}_{mode_str}.keras"
        scaler_path = f"{self.model_dir}/scaler_{mode_str}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path): 
            return None, f"Model or Scaler missing in '{self.model_dir}'! Please train them first."

        model = load_model(model_path, safe_mode=False)
        scaler = joblib.load(scaler_path)
        
        raw_inputs = np.array([[F, S, xcf, i+1] for i in range(20)])
        scaled_inputs = scaler.transform(raw_inputs)
        
        if model_name == "ann": removals = model.predict(scaled_inputs, verbose=0).flatten()[:N].tolist()
        else: removals = model.predict(scaled_inputs.reshape(1, 20, 4), verbose=0)[0, :N, 0].tolist()

        x_stages = [xcf * (1 - r/100) for r in removals]
        return (x_stages, [self.equilibrium(x) for x in x_stages], removals), "Success"

    # ==========================================
    # PLOTTING FUNCTIONS
    # ==========================================
    def plot_equilibrium(self):
        ax = self.canvas_eq.axes
        ax.clear()
        ax.plot(self.xB, self.xC, 'bo', markersize=6, label='Raffinate Phase')
        ax.plot(self.yB, self.yC, 'ro', markersize=6, label='Extract Phase')
        B_combined = np.concatenate([self.xB, self.yB[::-1]])
        C_combined = np.concatenate([self.xC, self.yC[::-1]])
        dome_interp = interp1d(B_combined, C_combined, kind='cubic')
        B_smooth = np.linspace(min(B_combined), max(B_combined), 200)
        C_smooth = dome_interp(B_smooth)
        ax.plot(B_smooth, C_smooth, 'k-', linewidth=2, alpha=0.6, label='Cubic Binodal Curve')
        for i in range(len(self.xB)):
            if i == 0: ax.plot([self.xB[i], self.yB[i]], [self.xC[i], self.yC[i]], 'g--', alpha=0.4, label='Tie Lines')
            else: ax.plot([self.xB[i], self.yB[i]], [self.xC[i], self.yC[i]], 'g--', alpha=0.4)
        ax.set_title("Liquid-Liquid Equilibrium Binodal Curve", fontsize=16, fontweight='bold')
        ax.set_xlabel("Solvent Fraction (B)", fontsize=12)
        ax.set_ylabel("Solute Fraction (C)", fontsize=12)
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=10)
        self.canvas_eq.draw()

    def plot_comparison(self, F, S, xcf, N):
        ax = self.canvas_comp.axes
        ax.clear()
        x_cross, _ = self.solve_crosscurrent(F, S, xcf, N)
        x_count, _ = self.solve_countercurrent(F, S, xcf, N)
        ax.plot(range(1, N + 1), [((xcf - x)/xcf)*100 for x in x_cross], 'go-', linewidth=2, label='Cross-current')
        ax.plot(range(1, N + 1), [((xcf - x)/xcf)*100 for x in x_count], 'mo-', linewidth=2, label='Counter-current')
        ax.set_title(f"{N}-Stage Performance Comparison")
        ax.set_xlabel("Number of Stages")
        ax.set_ylabel("Total Removal (%)")
        ax.set_xticks(range(1, N + 1))
        ax.grid(True, linestyle='--')
        ax.legend()
        self.canvas_comp.draw()

    def plot_heatmap(self, N, x_stages, y_stages, removals):
        ax = self.canvas_heat.axes
        ax.clear()
        data_matrix = np.array([x_stages, y_stages, removals]).T
        color_matrix = np.zeros_like(data_matrix)
        for j in range(3):
            c_min, c_max = data_matrix[:, j].min(), data_matrix[:, j].max()
            color_matrix[:, j] = (data_matrix[:, j] - c_min) / (c_max - c_min) if c_max > c_min else 0.5 
        cax = ax.imshow(color_matrix, aspect='auto', cmap='YlOrRd')
        for i in range(N):
            for j in range(3):
                val = data_matrix[i, j]
                txt_color = "white" if color_matrix[i, j] > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}%" if j == 2 else f"{val:.4f}", ha="center", va="center", color=txt_color, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Raffinate (x)", "Extract (y)", "Total Removal (%)"])
        ax.set_yticks(range(N))
        ax.set_yticklabels([f"Stage {i+1}" for i in range(N)])
        ax.set_title("Stage-by-Stage Heatmap (Column-Normalized)")
        if hasattr(self.canvas_heat, 'cbar') and self.canvas_heat.cbar:
            try: self.canvas_heat.cbar.remove(); self.canvas_heat.cbar = None
            except: pass
        self.canvas_heat.draw()

    def plot_3d_surface(self, F, S, xcf_max, is_cross, user_N):
        ax = self.canvas_3d.axes
        ax.clear()
        
        # 1. Generate the base 3D Surface
        stages = np.arange(1, max(2, user_N) + 1)
        feed_conc = np.linspace(0.05, max(0.1, xcf_max * 1.2), 15) 
        X, Y = np.meshgrid(stages, feed_conc)
        Z = np.zeros_like(X, dtype=float)
        
        for i, xF in enumerate(feed_conc):
            for j, n_stages in enumerate(stages):
                x_out, _ = self.solve_crosscurrent(F, S, xF, n_stages) if is_cross else self.solve_countercurrent(F, S, xF, n_stages)
                Z[i, j] = min(100.0, ((xF - x_out[-1]) / xF) * 100)
                
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        #ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis')

        # ---------------------------------------------------------
        # 2. ELBOW METHOD OPTIMIZATION ALGORITHM
        # ---------------------------------------------------------
        # Evaluate 1 to 20 stages at the user's specific feed composition (xcf_max)
        opt_stages = np.arange(1, 21) 
        opt_removals = []
        for n_stages in opt_stages:
            x_out, _ = self.solve_crosscurrent(F, S, xcf_max, n_stages) if is_cross else self.solve_countercurrent(F, S, xcf_max, n_stages)
            opt_removals.append(min(100.0, ((xcf_max - x_out[-1]) / xcf_max) * 100))
        
        # Normalize the data (Crucial, because Stages are 1-20 but Removal is 0-100)
        x_norm = (opt_stages - opt_stages[0]) / (opt_stages[-1] - opt_stages[0])
        y_min, y_max = min(opt_removals), max(opt_removals)
        y_norm = np.array([(y - y_min) / (y_max - y_min) if y_max > y_min else 0 for y in opt_removals])

        # Find the point with the maximum distance from the line connecting first and last points
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        
        max_dist = -1
        optimal_idx = 0
        
        for i in range(len(opt_stages)):
            p3 = np.array([x_norm[i], y_norm[i]])
            # Calculate perpendicular distance from point p3 to line (p1, p2)
            dist = np.abs(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1)
            if dist > max_dist:
                max_dist = dist
                optimal_idx = i
                
        optimal_n = opt_stages[optimal_idx]
        optimal_z = opt_removals[optimal_idx]

        # Plot the optimal point on the 3D surface
        ax.scatter([optimal_n], [xcf_max], [optimal_z], color='red', s=150, zorder=5, edgecolors='black', label=f'Optimal point (Elbow N={optimal_n})')
        ax.legend(loc='upper left')
        # ---------------------------------------------------------

        ax.set_zlim(0, 100)
        ax.view_init(elev=25, azim=-45)
        ax.set_title(f"3D Response Surface: {'Cross-current' if is_cross else 'Counter-current'} Efficiency")
        ax.set_xlabel("Number of Stages (N)")
        ax.set_ylabel("Feed Composition (xcf)")
        ax.set_zlabel("Total Removal (%)")
        ax.set_xticks(stages)
        
        if not hasattr(self.canvas_3d, 'cbar') or self.canvas_3d.cbar is None: 
            self.canvas_3d.cbar = self.canvas_3d.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        else: 
            self.canvas_3d.cbar.update_normal(surf)
            
        self.canvas_3d.draw()

    def plot_dynamic_contour(self, F, xcf):
        self.canvas_contour.setFixedSize(950, 4500) 
        self.canvas_contour.fig.clf() 
        axs = self.canvas_contour.fig.subplots(8, 1)
        self.canvas_contour.fig.subplots_adjust(hspace=0.8, left=0.12, right=0.85, bottom=0.04, top=0.94)
        
        solvent_range = np.linspace(100, 5000, 30) 
        stage_range = np.arange(1, 21)             
        X, Y = np.meshgrid(stage_range, solvent_range)

        models_to_plot = [
            ("ann", True, 0), ("rnn", True, 1), ("lstm", True, 2), ("gru", True, 3),
            ("ann", False, 4), ("rnn", False, 5), ("lstm", False, 6), ("gru", False, 7)
        ]

        for mod_name, is_cross, idx in models_to_plot:
            ax = axs[idx]
            mode_str = "crosscurrent" if is_cross else "countercurrent"
            
            # USE DYNAMIC DIRECTORY
            model_path = f"{self.model_dir}/{mod_name}_{mode_str}.keras"
            scaler_path = f"{self.model_dir}/scaler_{mode_str}.pkl"
            
            title = f"{mod_name.upper()} ({'Cross' if is_cross else 'Counter'}-Current)"
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                ax.text(0.5, 0.5, f"Missing\n{mod_name}", ha='center', va='center', fontsize=14)
                ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
                ax.axis('off')
                continue

            try:
                model = load_model(model_path, safe_mode=False)
                scaler = joblib.load(scaler_path)
            except Exception:
                ax.text(0.5, 0.5, "Load Error", ha='center', va='center', fontsize=14)
                ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
                ax.axis('off')
                continue

            if mod_name == "ann":
                batch_X = [[F, S_val, xcf, stg] for S_val in solvent_range for stg in stage_range]
                preds = model.predict(scaler.transform(np.array(batch_X)), verbose=0).flatten()
                Z = preds.reshape(len(solvent_range), len(stage_range))
            else:
                batch_X_seq = [[[F, S_val, xcf, stg] for stg in range(1, 21)] for S_val in solvent_range]
                scaled_seq = scaler.transform(np.array(batch_X_seq).reshape(-1, 4)).reshape(-1, 20, 4)
                Z = model.predict(scaled_seq, verbose=0)[:, :, 0]

            cp = ax.contourf(X, Y, Z, levels=25, cmap='viridis')
            ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
            ax.set_xticks(np.arange(2, 21, 2)) 
            ax.set_xlabel('Stages (N)', fontsize=14)
            ax.set_ylabel('Solvent (S)', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)

            cbar = self.canvas_contour.fig.colorbar(cp, ax=ax, orientation='vertical', pad=0.03, aspect=15)
            cbar.set_label('Removal (%)', fontsize=14, fontweight='bold')
        
        self.canvas_contour.fig.suptitle(f"Live AI Response Contours (Feed = {F}, xcf = {xcf})", fontsize=14, fontweight='bold', y=0.98)
        self.canvas_contour.axes = axs[0] 
        self.canvas_contour.draw()

    # ==========================================
    # MASTER EXECUTION
    # ==========================================
    def run_calculations(self):
        F, S, xcf, N = self.input_F.value(), self.input_S.value(), self.input_xcf.value(), self.input_N.value()
        method, is_cross = self.combo_method.currentText(), self.radio_cross.isChecked()
        
        self.btn_run.setText("Calculating... Please wait.")
        QApplication.processEvents() 

        if "AI" in method:
            if not AI_AVAILABLE:
                QMessageBox.critical(self, "Error", "TensorFlow is not installed!")
                self.btn_run.setText("RUN SIMULATION")
                return
            
            ai_data, msg = self.solve_ai_model(F, S, xcf, N, is_cross, method)
            if ai_data is None:
                QMessageBox.warning(self, "Missing AI Models", msg)
                self.btn_run.setText("RUN SIMULATION")
                return
            x_stages, y_stages, removals = ai_data
            title_tag = f"[AI PREDICTION: {method.split(': ')[1]}]"
        else:
            x_stages, y_stages = self.solve_crosscurrent(F, S, xcf, N) if is_cross else self.solve_countercurrent(F, S, xcf, N)
            removals = [((xcf - x_stages[i]) / xcf) * 100 for i in range(N)]
            title_tag = "[RIGOROUS MATH (fsolve)]"

        self.lbl_result.setText(f"{title_tag}\nFinal {'Cross-current' if is_cross else 'Counter-current'} Removal: {removals[-1]:.2f}%")

        self.table_results.setRowCount(N)
        for i in range(N):
            self.table_results.setItem(i, 0, QTableWidgetItem(f"Stage {i+1}"))
            self.table_results.setItem(i, 1, QTableWidgetItem(f"{x_stages[i]:.4f}"))
            self.table_results.setItem(i, 2, QTableWidgetItem(f"{y_stages[i]:.4f}"))
            self.table_results.setItem(i, 3, QTableWidgetItem(f"{removals[i]:.2f}%"))

        self.plot_comparison(F, S, xcf, N)
        self.plot_heatmap(N, x_stages, y_stages, removals)
        self.plot_3d_surface(F, S, xcf, is_cross, N)
        self.plot_dynamic_contour(F, xcf)
        
        self.btn_run.setText("RUN SIMULATION")
        self.tabs.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExtractionApp()
    window.show()
    sys.exit(app.exec_())