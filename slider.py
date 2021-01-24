from tkinter import *

def sel_accx(i):
   selection = "Value = " + str(varx.get())
   label.config(text=selection)

def sel_accy(i):
   selection = "Value = " + str(vary.get())
   label.config(text=selection)

root = Tk()
varx = DoubleVar()
vary = DoubleVar()
scale_accx1 = Scale(root, variable=varx, command=sel_accx)
scale_accx1.pack(anchor=CENTER)

scale_accy1 = Scale(root, variable=vary, command=sel_accy)
scale_accy1.pack(anchor=CENTER)

# button = Button(root, text="Get Scale Value", command=sel)
# button.pack(anchor=CENTER)

label = Label(root)
label.pack()

root.mainloop()