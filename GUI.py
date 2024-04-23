import customtkinter
import os
from PIL import Image
import tkinter.filedialog as filedialog
import tkinter as tk
import core
from io import StringIO
import sys
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

        self.model_button = customtkinter.CTkButton(self.navigation_frame,
                                                      corner_radius=0,
                                                      height=40,
                                                      border_spacing=10,
                                                      text="Model",
                                                      fg_color="transparent",
                                                      text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      font=customtkinter.CTkFont(size=20),
                                                      command=self.model_button_event)
        self.model_button.grid(row=3, column=0, sticky="ew")

        self.view_button = customtkinter.CTkButton(self.navigation_frame,
                                                      corner_radius=0,
                                                      height=40,
                                                      border_spacing=10,
                                                      text="View",
                                                      fg_color="transparent",
                                                      text_color=(
                                                      "gray10", "gray90"),
                                                      hover_color=(
                                                      "gray70", "gray30"),
                                                      font=customtkinter.CTkFont(size=20),
                                                      command=self.view_button_event)
        self.view_button.grid(row=4, column=0, sticky="ew")

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

        self.model_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.view_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")




        self.frame_change("dataset")


    def frame_change(self, name):
        self.dataset_button.configure(fg_color=("gray75", "gray25") if name == "dataset" else "transparent")
        self.processing_button.configure(fg_color=("gray75", "gray25") if name == "processing" else "transparent")
        self.model_button.configure(fg_color=("gray75", "gray25") if name == "model" else "transparent")
        self.view_button.configure(fg_color=("gray75", "gray25") if name == "view" else "transparent")
        if name == "dataset":
            self.dataset_frame.grid(row=0, column=1, sticky="nsew")
            if self.current_frame == 2:
                buffer = StringIO()
                sys.stdout = buffer
                print(self.dataset[0])
                sys.stdout = sys.__stdout__
                df_string = buffer.getvalue()
                self.dataset_preview.delete("0.0", "end")
                self.dataset_preview.insert("0.0", df_string)
            self.current_frame = 1
        else:
            self.dataset_frame.grid_forget()
        if name == "processing":
            self.processing_frame.grid(row=0, column=1, sticky="nsew")
            self.current_frame = 2
            buffer = StringIO()
            sys.stdout = buffer
            print(self.dataset[0])
            sys.stdout = sys.__stdout__
            df_string = buffer.getvalue()
            self.dataset_preview2.delete("0.0", "end")
            self.dataset_preview2.insert("0.0", df_string)
        else:
            self.processing_frame.grid_forget()
        if name == "model":
            self.model_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.model_frame.grid_forget()
        if name == "view":
            self.view_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.view_frame.grid_forget()


    def dataset_button_event(self):
        self.frame_change("dataset")

    def processing_button_event(self):
        self.frame_change("processing")

    def model_button_event(self):
        self.frame_change("model")

    def view_button_event(self):
        self.frame_change("view")

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
            
            
    #pca funguje, fixnout chybějící datasrt později







if __name__ == "__main__":
    app = App()
    app.mainloop()
