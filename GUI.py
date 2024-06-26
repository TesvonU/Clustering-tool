import customtkinter
import os
from PIL import Image
import tkinter.filedialog as filedialog
import tkinter as tk
import core
from io import StringIO
import sys
import numpy as np
import pandas as pd

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        self.title("Clustering tool.py")
        self.geometry("700x450")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.dataset = None
        self.current_frame = 1
        self.model_dict = {
            'KMeans': {'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'random_state': 42},
            'AgglomerativeClustering': {'n_clusters': 3, 'affinity': 'euclidean', 'linkage': 'ward'},
            'DBSCAN': {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'},
            'MeanShift': {'bandwidth': 0.5, 'seeds': None, 'bin_seeding': False, 'cluster_all': True, 'min_bin_freq': 1, 'max_iter': 300},
            'GaussianMixture': {'n_components': 3, 'covariance_type': 'full', 'tol': 1e-3, 'max_iter': 100, 'n_init': 1, 'init_params': 'kmeans', 'random_state': 42},
            'Birch': {'threshold': 0.5, 'branching_factor': 50, 'n_clusters': 3, 'compute_labels': True},
            'AffinityPropagation': {'damping': 0.5, 'max_iter': 200, 'convergence_iter': 15, 'preference': None, 'affinity': 'euclidean'},
            'SpectralClustering': {'n_clusters': 3, 'eigen_solver': None, 'random_state': None, 'n_init': 10, 'gamma': 1.0, 'n_neighbors': 10, 'eigen_tol': 0.0, 'assign_labels': 'kmeans'},
            'OPTICS': {'min_samples': 5, 'max_eps': np.inf, 'metric': 'minkowski', 'p': 2, 'cluster_method': 'xi', 'eps': None, 'xi': 0.05, 'predecessor_correction': True, 'min_cluster_size': None},
                            }

        #Image paths
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"test_images")
        self.logo = customtkinter.CTkImage(Image.open(os.path.join(image_path, "cluster.png")),size=(26, 26))
        self.folder = customtkinter.CTkImage(Image.open(os.path.join(image_path, "folder.png")), size=(26, 26))
        self.downloadicon = customtkinter.CTkImage(Image.open(os.path.join(image_path, "download.png")), size=(26, 26))
        #navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(5, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame,
                                                             text="Clustering tool",
                                                             image=self.logo,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.dataset_button = customtkinter.CTkButton(self.navigation_frame,
                                                   corner_radius=0, height=40,
                                                   border_spacing=10,
                                                   text="Dataset",
                                                   fg_color="transparent",
                                                   text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   font=customtkinter.CTkFont(size=20),
                                                   command=self.dataset_button_event)
        self.dataset_button.grid(row=1, column=0, sticky="ew")

        self.processing_button = customtkinter.CTkButton(self.navigation_frame,
                                                      corner_radius=0,
                                                      height=40,
                                                      border_spacing=10,
                                                      text="Pre-proccesing",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      font=customtkinter.CTkFont(size=20),
                                                      command=self.processing_button_event)
        self.processing_button.grid(row=2, column=0, sticky="ew")

        self.anomaly_button = customtkinter.CTkButton(self.navigation_frame,
                                                      corner_radius=0,
                                                      height=40,
                                                      border_spacing=10,
                                                      text="Anomaly removal",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      font=customtkinter.CTkFont(size=20),
                                                      command=self.anomaly_button_event)
        self.anomaly_button.grid(row=3, column=0, sticky="ew")

        self.model_button = customtkinter.CTkButton(self.navigation_frame,
                                                      corner_radius=0,
                                                      height=40,
                                                      border_spacing=10,
                                                      text="Model",
                                                      fg_color="transparent",
                                                      text_color=(
                                                      "gray10", "gray90"),
                                                      hover_color=(
                                                      "gray70", "gray30"),
                                                      font=customtkinter.CTkFont(size=20),
                                                      command=self.model_button_event)
        self.model_button.grid(row=4, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            self.navigation_frame, values=["Dark", "Light"],
            command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=5, column=0, padx=20, pady=20, sticky="s")

        #frame 1
        self.dataset_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.dataset_frame.grid_rowconfigure(3, weight=2)
        self.dataset_frame.grid_columnconfigure(3, weight=2)
        self.dataset_frame_lower = customtkinter.CTkFrame(self.dataset_frame, corner_radius=0,fg_color="transparent")
        self.dataset_frame_lower.grid(row=4, column=0)
        self.dataset_frame.grid_rowconfigure(1, weight=3)
        self.dataset_frame.grid_columnconfigure(3, weight=3)
        self.dataset_entry = customtkinter.CTkEntry(self.dataset_frame_lower, placeholder_text="path to Dataset", width=350, height=35, font=customtkinter.CTkFont(size=15))
        self.dataset_entry.grid(row=0, column=0, pady=20, padx=10, sticky="w")
        self.dataset_button = customtkinter.CTkButton(self.dataset_frame_lower, image=self.folder , command=self.openfile, text="", width=10, height=20, border_color="gray40", border_width=2)
        self.dataset_button.grid(row=0, column=1, pady=20, padx=0, sticky="w")
        self.save_button2 = customtkinter.CTkButton(self.dataset_frame_lower, image=self.downloadicon, command=self.save_file, text="", width=10, height=20)
        self.save_button2.grid(row=0, column=2, pady=20, padx=10, sticky="w")
        self.dataset_frame_middle = customtkinter.CTkFrame(self.dataset_frame, corner_radius=0,fg_color="transparent")
        self.dataset_frame_middle.grid(row=1, column=0)
        self.dataset_frame_middle.grid_rowconfigure(3, weight=3)
        self.dataset_frame_middle.grid_columnconfigure(3, weight=3)
        self.row_from = customtkinter.CTkEntry(self.dataset_frame_middle, placeholder_text="0000", font=customtkinter.CTkFont(size=15))
        self.row_from.grid(row=0, column=0, padx=10, pady=5)
        self.row_to = customtkinter.CTkEntry(self.dataset_frame_middle, placeholder_text="0000", font=customtkinter.CTkFont(size=15))
        self.row_to.grid(row=0, column=1, padx=10, pady=5)
        self.row_button = customtkinter.CTkButton(self.dataset_frame_middle, text="Cut rows", font=customtkinter.CTkFont(size=15), command=self.cut_rows)
        self.row_button.grid(row=0, column=2, padx=10, pady=5)
        self.col_from = customtkinter.CTkEntry(self.dataset_frame_middle, placeholder_text="Column name", font=customtkinter.CTkFont(size=15))
        self.col_from.grid(row=1, column=0, padx=10, pady=5)
        self.col_button = customtkinter.CTkButton(self.dataset_frame_middle, text="Drop columns", font=customtkinter.CTkFont(size=15), command=self.cut_columns)
        self.col_button.grid(row=1, column=2, padx=10, pady=5)
        self.sort_button = customtkinter.CTkButton(self.dataset_frame_middle, text="Sort by", font=customtkinter.CTkFont(size=15), command=self.sort_by)
        self.sort_button.grid(row=1, column=1, padx=10, pady=5)
        self.dataset_preview = customtkinter.CTkTextbox(self.dataset_frame, width=430, wrap="none")
        self.dataset_preview.grid(row=0, column=0, pady=20, padx=50)

        #frame 2
        self.processing_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.processing_frame.grid_rowconfigure(3, weight=2)
        self.processing_frame.grid_columnconfigure(3, weight=2)
        self.dataset_preview2 = customtkinter.CTkTextbox(self.processing_frame, width=430, wrap="none")
        self.dataset_preview2.grid(row=0, column=0, pady=20, padx=50)
        self.processing_frame_middle = customtkinter.CTkFrame(self.processing_frame, corner_radius=0,fg_color="transparent")
        self.processing_frame_middle.grid(row=1, column=0)
        self.processing_frame_middle.grid_rowconfigure(3, weight=3)
        self.processing_frame_middle.grid_columnconfigure(3, weight=3)
        self.strategy_entry = customtkinter.CTkEntry(self.processing_frame_middle, placeholder_text="KNN/Simple", font=customtkinter.CTkFont(size=13))
        self.strategy_entry.grid(row=0, column=0, padx=10, pady=5)
        self.impute_button = customtkinter.CTkButton(self.processing_frame_middle, text="Impute NaNs", font=customtkinter.CTkFont(size=15), command=self.impute)
        self.impute_button.grid(row=0, column=1, padx=10, pady=5)
        self.unique_entry = customtkinter.CTkEntry(self.processing_frame_middle, placeholder_text="Unique value", font=customtkinter.CTkFont(size=13))
        self.unique_entry.grid(row=1, column=0, padx=10, pady=5)
        self.duplicate_button = customtkinter.CTkButton(self.processing_frame_middle, text="Remove duplicates", font=customtkinter.CTkFont(size=15), command=self.remove_duplicates)
        self.duplicate_button.grid(row=1, column=1, padx=10, pady=5)
        self.scale_button = customtkinter.CTkButton(self.processing_frame_middle, text="Scale", font=customtkinter.CTkFont(size=15), command=self.scale)
        self.scale_button.grid(row=3, column=1, padx=10, pady=5)
        self.dimension_entry = customtkinter.CTkEntry(self.processing_frame_middle, placeholder_text="0", font=customtkinter.CTkFont(size=13))
        self.dimension_entry.grid(row=2, column=0, padx=10, pady=5)
        self.pca_button = customtkinter.CTkButton(self.processing_frame_middle, text="PCA reduction", font=customtkinter.CTkFont(size=15), command=self.PCA)
        self.pca_button.grid(row=2, column=1, padx=10, pady=5)

        #frame 3
        self.anomaly_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.anomaly_frame.grid_rowconfigure(3, weight=2)
        self.anomaly_frame.grid_columnconfigure(3, weight=2)
        self.dataset_preview3 = customtkinter.CTkTextbox(self.anomaly_frame, width=430, wrap="none")
        self.dataset_preview3.grid(row=0, column=0, pady=20, padx=50)
        self.anomaly_frame_middle = customtkinter.CTkFrame(self.anomaly_frame, corner_radius=0, fg_color="transparent")
        self.anomaly_frame_middle.grid(row=1, column=0)
        self.anomaly_frame_middle.grid_rowconfigure(3, weight=3)
        self.anomaly_frame_middle.grid_columnconfigure(3, weight=3)
        self.global_percentage_entry = customtkinter.CTkEntry(self.anomaly_frame_middle, placeholder_text="0.3", font=customtkinter.CTkFont(size=15))
        self.global_percentage_entry.grid(row=0, column=0, padx=10, pady=5)
        self.global_percentage_button = customtkinter.CTkButton(self.anomaly_frame_middle, text="  Same contamination   ", font=customtkinter.CTkFont(size=15), command=self.remove_anomaly)
        self.global_percentage_button.grid(row=0, column=1, padx=10, pady=5)
        self.differnt_percentage_entry = customtkinter.CTkEntry(self.anomaly_frame_middle, placeholder_text="0.3;0.4;0.5;0.26...", font=customtkinter.CTkFont(size=15))
        self.differnt_percentage_entry.grid(row=1, column=0, padx=10, pady=5)
        self.differnt_percentage = customtkinter.CTkButton(self.anomaly_frame_middle, text="Different contamination", font=customtkinter.CTkFont(size=15), command=self.remove_anomaly2)
        self.differnt_percentage.grid(row=1, column=1, padx=10, pady=5)
        #frame4
        self.model_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.model_frame.grid_rowconfigure(3, weight=2)
        self.model_frame.grid_columnconfigure(3, weight=2)
        self.model_settings = customtkinter.CTkTextbox(self.model_frame, width=430, wrap="none", font=customtkinter.CTkFont(size=15))
        self.model_settings.grid(row=0, column=0, pady=20, padx=50)
        self.model_settings.insert("0.0", "NO MODEL SELECTED")
        self.model_frame_middle = customtkinter.CTkFrame(self.model_frame, corner_radius=0, fg_color="transparent")
        self.model_frame_middle.grid(row=1, column=0)
        self.model_frame_middle.grid_rowconfigure(3, weight=3)
        self.model_frame_middle.grid_columnconfigure(3, weight=3)
        self.model_selection = customtkinter.CTkComboBox(self.model_frame_middle, values=[key for key in self.model_dict], command=self.update_model_settings, width=200)
        self.model_selection.grid(row=0, column=2, padx=10, pady=10)
        self.update_model_settings(0)
        self.label_button = customtkinter.CTkButton(self.model_frame_middle, text="Create Labels", font=customtkinter.CTkFont(size=15), command=self.create_labels)
        self.label_button.grid(row=0, column=1, padx=10, pady=10)


        self.frame_change("dataset")

    def frame_change(self, name):
        self.dataset_button.configure(fg_color=("gray75", "gray25") if name == "dataset" else "transparent")
        self.processing_button.configure(fg_color=("gray75", "gray25") if name == "processing" else "transparent")
        self.anomaly_button.configure(fg_color=("gray75", "gray25") if name == "anomaly" else "transparent")
        self.model_button.configure(fg_color=("gray75", "gray25") if name == "model" else "transparent")
        if name == "dataset":
            self.dataset_frame.grid(row=0, column=1, sticky="nsew")
            if self.dataset:
                buffer = StringIO()
                sys.stdout = buffer
                print(self.dataset[0])
                sys.stdout = sys.__stdout__
                df_string = buffer.getvalue()
                self.dataset_preview.delete("0.0", "end")
                self.dataset_preview.insert("0.0", df_string)
        else:
            self.dataset_frame.grid_forget()
        if name == "processing":
            self.processing_frame.grid(row=0, column=1, sticky="nsew")
            if self.dataset:
                buffer = StringIO()
                sys.stdout = buffer
                print(self.dataset[0])
                sys.stdout = sys.__stdout__
                df_string = buffer.getvalue()
                self.dataset_preview2.delete("0.0", "end")
                self.dataset_preview2.insert("0.0", df_string)
        else:
            self.processing_frame.grid_forget()
        if name == "anomaly":
            self.anomaly_frame.grid(row=0, column=1, sticky="nsew")
            if self.dataset:
                buffer = StringIO()
                sys.stdout = buffer
                print(self.dataset[0])
                sys.stdout = sys.__stdout__
                df_string = buffer.getvalue()
                self.dataset_preview3.delete("0.0", "end")
                self.dataset_preview3.insert("0.0", df_string)
        else:
            self.anomaly_frame.grid_forget()
        if name == "model":
            self.model_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_frame.grid_forget()


    def dataset_button_event(self):
        self.frame_change("dataset")

    def processing_button_event(self):
        self.frame_change("processing")

    def anomaly_button_event(self):
        self.frame_change("anomaly")

    def model_button_event(self):
        self.frame_change("model")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def openfile(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.dataset_entry.delete(0, 'end')  # Clear any existing text
            self.dataset_entry.insert(0, filepath)  # Insert the selected file path
            self.dataset = core.read_file(filepath)
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset)
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview.delete("0.0", "end")
            self.dataset_preview.insert("0.0", df_string)

    def save_file(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            core.save_dataset(self.dataset[0], filepath)

    def cut_rows(self):
        row_from = self.row_from.get()
        row_to = self.row_to.get()
        self.dataset = core.drop_lines(self.dataset[0], row_from, row_to)
        self.row_to.delete(0, "end")
        self.row_to.insert(0, "0")
        self.row_from.delete(0, "end")
        self.row_from.insert(0, "0")
        buffer = StringIO()
        sys.stdout = buffer
        print(self.dataset[0])
        sys.stdout = sys.__stdout__
        df_string = buffer.getvalue()
        self.dataset_preview.delete("0.0", "end")
        self.dataset_preview.insert("0.0", df_string)

    def cut_columns(self):
        col_from = self.col_from.get()
        self.dataset = core.drop_column(self.dataset[0], col_from )
        self.col_from.delete(0, "end")
        buffer = StringIO()
        sys.stdout = buffer
        print(self.dataset[0])
        sys.stdout = sys.__stdout__
        df_string = buffer.getvalue()
        self.dataset_preview.delete("0.0", "end")
        self.dataset_preview.insert("0.0", df_string)

    def sort_by(self):
        sort_column = self.col_from.get()
        self.col_from.delete(0, "end")
        self.dataset = core.sort_dataset(self.dataset[0], sort_column)
        buffer = StringIO()
        sys.stdout = buffer
        print(self.dataset[0])
        sys.stdout = sys.__stdout__
        df_string = buffer.getvalue()
        self.dataset_preview.delete("0.0", "end")
        self.dataset_preview.insert("0.0", df_string)

    def remove_duplicates(self):
        value = self.unique_entry.get()
        self.unique_entry.delete(0, "end")
        self.dataset = core.drop_duplicates(self.dataset[0], value)
        buffer = StringIO()
        sys.stdout = buffer
        print(self.dataset[0])
        sys.stdout = sys.__stdout__
        df_string = buffer.getvalue()
        self.dataset_preview2.delete("0.0", "end")
        self.dataset_preview2.insert("0.0", df_string)

    def impute(self):
        strategy = self.strategy_entry.get()
        self.strategy_entry.delete(0, "end")

        answer = core.inpute_nan(self.dataset[0], strategy)
        if len(answer) == 4:
            self.dataset_preview2.insert("0.0", "REMOVE STRINGS FIRST\n")
            self.strategy_entry.delete(0, "end")
            self.strategy_entry.insert(0, "Remove strings first")
        else:
            self.dataset = answer
            buffer = StringIO()
            sys.stdout = buffer
            if self.dataset is not tuple:
                print(self.dataset)
            else:
                print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview2.delete("0.0", "end")
            self.dataset_preview2.insert("0.0", df_string)

    def scale(self):
        answer = core.scale_dataset(self.dataset[0])
        if len(answer) == 4:
            self.dataset_preview2.insert("0.0", "REMOVE STRINGS FIRST\n")
        else:
            self.dataset = answer
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview2.delete("0.0", "end")
            self.dataset_preview2.insert("0.0", df_string)

    def PCA(self):
        dimensions = self.dimension_entry.get()
        self.dimension_entry.delete(0, "end")
        answer = core.pca_reduction(self.dataset[0], dimensions)
        if len(answer) == 4:
            self.dataset_preview2.insert("0.0", "REMOVE STRINGS AND NaNs FIRST\n")
        else:
            self.dataset = answer
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview2.delete("0.0", "end")
            self.dataset_preview2.insert("0.0", df_string)

    def update_model_settings(self, _):
        selected = self.model_selection.get()
        if selected in self.model_dict:
            params = self.model_dict[selected]
            self.model_settings.delete("1.0", "end")
            for param, value in params.items():
                self.model_settings.insert("end", f"{param}: {value}\n")
            self.model_settings.insert("end", "\nParameter description can be found in documentation.\n")
            print(selected)
            if selected in ["AgglomerativeClustering", "AffinityPropagation", "SpectralClustering"]:
                self.model_settings.insert("end", "Model is not optimal for larger datasets.")



    def remove_anomaly(self):
        percentage = self.global_percentage_entry.get()
        if ";" in percentage:
            percentage = percentage.split(";")
        else:
            if percentage == "":
                percentage = "0.3"
            percentage = [percentage]
        print(percentage)
        self.global_percentage_entry.delete(0, "end")
        answer = core.remove_anomalies(self.dataset[0], percentage)
        print("ans", answer)
        if len(answer) == 4:
            self.dataset_preview2.insert("0.0",
                                         "REMOVE STRINGS AND NaNs FIRST\n")
        else:
            self.dataset = answer
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview3.delete("0.0", "end")
            self.dataset_preview3.insert("0.0", df_string)


    def remove_anomaly2(self):
        percentage = self.differnt_percentage_entry.get()
        if ";" in percentage:
            percentage = percentage.split(";")
        else:
            if percentage == "":
                percentage = "0.3"
            percentage = [percentage]
        print(percentage)
        self.differnt_percentage_entry.delete(0, "end")
        answer = core.remove_anomalies(self.dataset[0], percentage)
        print("ans", answer)
        if len(answer) == 4:
            self.dataset_preview2.insert("0.0",
                                         "REMOVE STRINGS AND NaNs FIRST\n")
        else:
            self.dataset = answer
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview3.delete("0.0", "end")
            self.dataset_preview3.insert("0.0", df_string)

    def create_labels(self):
        model_name = self.model_selection.get()
        if model_name in self.model_dict:
            model_params_text = self.model_settings.get("1.0", "end")
            model_params_lines = model_params_text.split("\n")
            model_params = {}
            for line in model_params_lines:
                param_value = line.split(":")
                if len(param_value) == 2:
                    param = param_value[0].strip()
                    value = param_value[1].strip()
                    model_params[param] = value
            result = core.run_model(model_name, model_params, self.dataset[0])
            print("Labels:", result)
            result_df = pd.DataFrame({"ID": self.dataset[0]["ID"], "Label": result})
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV files",
                                                                 "*.csv")])
            if file_path:
                result_df.to_csv(file_path, index=False)
                print("Data saved")
            else:
                print("No file selected")
        else:
            print("Invalid model name:", model_name)


    #pca funguje, fixnout chybějící datasrt později







if __name__ == "__main__":
    app = App()
    app.mainloop()
