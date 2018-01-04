from Tkinter import *

fields = 'Cell Type', 'Cell Subtype'
cell_type = ""
cell_subtype = ""

result_file = "result.txt"
image_name = "2.png"

def fetch(entries):
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        print('%s: "%s"' %(field, text))

def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=15, text=field, anchor='w')
        ent = Entry(row)

        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)

        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))

    return entries

def save_and_quit(image_name, ents, root):
    global cell_type, cell_subtype

    with open(result_file, 'ab') as f:
        f.write(image_name)
        for ent in ents:
            (field, ans) = ent
            print(ans.get())
            f.write(","+ans.get())

        f.write("\n")

    root.destroy()

def enter_label(image_name):
    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, im=image_name, e=ents, r=root: save_and_quit(im, e, r)))

    b2 = Button(root, text='Quit', command=root.quit)
    b2.pack(side=RIGHT, padx=5, pady=5)

    b1 = Button(root, text='Enter', command=(lambda: save_and_quit(image_name, ents, root)))
    b1.pack(side=RIGHT, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, im=image_name, e=ents, r=root: save_and_quit(im, e, r)))

    b2 = Button(root, text='Quit', command=root.quit)
    b2.pack(side=RIGHT, padx=5, pady=5)

    b1 = Button(root, text='Enter', command=(lambda: save_and_quit(image_name, ents, root)))
    b1.pack(side=RIGHT, padx=5, pady=5)

    root.mainloop()
