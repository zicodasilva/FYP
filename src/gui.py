import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog
import os
from calib import calib, app, extract, utils, plotting
import plot as pt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        """
        Initialise the main application
        """
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Montserrat', size=18)
        self.normal_font = tkfont.Font(family='Montserrat', size=12)
        self.project_dir = "No project folder chosen"

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self, width=1130, height=720)
        nav_bar = tk.Frame(self, width=150, height=720, background="#2c3e50")
        container.pack_propagate(False)
        nav_bar.pack_propagate(False)
        nav_bar.place(relx=0, rely=0)
        container.place(x=150,y=0)
        self.home_label = tk.Label(nav_bar, text="Home", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.home_label.place(relx=0.5, y=30, anchor="center")
        self.build_label = tk.Label(nav_bar, text="Build", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.build_label.place(relx=0.5, y=80, anchor="center")
        self.analyse_label = tk.Label(nav_bar, text="Analyse", font=self.title_font, bg="#2c3e50", fg="#ffffff", width=150, cursor="hand2")
        self.analyse_label.place(relx=0.5, y=130, anchor="center")

        self.home_label.bind("<Enter>", self.home_on_enter)
        self.home_label.bind("<Leave>", self.home_on_leave)
        self.home_label.bind("<Button-1>", self.home)

        self.build_label.bind("<Enter>", self.build_on_enter)
        self.build_label.bind("<Leave>", self.build_on_leave)
        self.build_label.bind("<Button-1>", self.build)

        self.analyse_label.bind("<Enter>", self.analyse_on_enter)
        self.analyse_label.bind("<Leave>", self.analyse_on_leave)
        self.analyse_label.bind("<Button-1>", self.analyse)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.place(x=0,y=0)

        self.show_frame("StartPage")

    # Binding events to change label appearances on enter, exit, and click
    def home(self, event):
        self.show_frame("StartPage")

    def home_on_enter(self, event):
        self.home_label.configure(bg="#34495e")

    def home_on_leave(self, event):
        self.home_label.configure(bg="#2c3e50")

    def build(self, event):
        self.show_frame("PageOne")

    def build_on_enter(self, event):
        self.build_label.configure(bg="#34495e")

    def build_on_leave(self, event):
        self.build_label.configure(bg="#2c3e50")

    def analyse(self, event):
        self.show_frame("PageTwo")

    def analyse_on_enter(self, event):
        self.analyse_label.configure(bg="#34495e")

    def analyse_on_leave(self, event):
        self.analyse_label.configure(bg="#2c3e50")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the home page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, bg="#ffffff")
        self.controller = controller
        self.pack_propagate(False)
        label_folder = tk.Label(self, text=controller.project_dir, font=controller.normal_font, bg="#ffffff")
        label_folder.place(relx=0.5, rely=0.55, anchor="center")
        
        def choose_folder():
            currdir = os.getcwd()
            controller.project_dir = filedialog.askdirectory(parent=self, initialdir=currdir, title='Please Select a Project Folder:')
            if len(controller.project_dir) > 0:
                print("You chose %s" % controller.project_dir)
                label_folder.configure(text=controller.project_dir)
        
        label = tk.Label(self, text="Home", font=controller.title_font, bg="#ffffff")
        label.place(relx=0.5, rely=0.1, anchor="center")
        button1 = tk.Button(self, text="Choose Project Folder", command=choose_folder)
        button1.place(relx=0.5, rely=0.5, anchor="center")
        

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the build page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)

        def plot_points():
            label = tk.Label(self, text=controller.project_dir, font=controller.normal_font, background="#ffffff")
            label.place(relx=0.5, rely=0.5, anchor="center")
            f = Figure(figsize=(5,5), dpi=100)
            a = f.add_subplot(111, projection="3d")
            a.view_init(elev=20., azim=60)
            parts = ["l_eye", "r_eye", "spine", "l_back_paw", "r_back_paw"]
            for part in parts:
                vals = pt.plot_skeleton(controller.project_dir, part)
                a.scatter(vals[0][2],vals[1][2],vals[2][2])

            canvas = FigureCanvasTkAgg(f, self)
            canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
            canvas._tkcanvas.place(relx=0.5, rely=0.5, anchor="center")

        label = tk.Label(self, text="Build", font=controller.title_font, background="#ffffff")
        label.place(relx=0, rely=0)
        button = tk.Button(self, text="Plot", command=plot_points)
        button.place(relx=0.5, rely=0.1, anchor="center")

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        """
        Initialise a frame for the analyse page
        """
        tk.Frame.__init__(self, parent, height=720, width=1130, background="#ffffff")
        self.controller = controller
        self.pack_propagate(False)
        label = tk.Label(self, text="Analyse", font=controller.title_font, background="#ffffff")
        label.place(relx=0, rely=0)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()

if __name__ == "__main__":
    app = Application()
    app.geometry("1280x720")
    app.title("Final Year Project")
    app.mainloop()