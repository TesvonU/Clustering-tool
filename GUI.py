import customtkinter
import os
from PIL import Image

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        self.title("Clustering tool.py")
        self.geometry("700x450")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        #Image paths
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"test_images")
        self.logo = customtkinter.CTkImage(Image.open(os.path.join(image_path, "cluster.png")),size=(26, 26))

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

        #frames
        self.dataset_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.processing_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
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
        else:
            self.dataset_frame.grid_forget()
        if name == "processing":
            self.processing_frame.grid(row=0, column=1, sticky="nsew")
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








if __name__ == "__main__":
    app = App()
    app.mainloop()
