Labeller module
================================================================================================
Labeller module of CellNet

## Labler
This is the labling execution module for CellNet Test set labelling.
Upload your image in the execution file, drag the area, and type in the label of the cell sub-class.
The execution file will automatically convert the given information into a csv format.

## Requirements
- [PyGTK](http://www.pygtk.org/)
- [PyInStaller](http://www.pyinstaller.org/)

```bash
# PyGTK
sudo apt-get install python-gtk2

# PyInstaller
pip install pyinstaller
```

## Basic Setups
[TODO]

```bash
pyinstaller main.py # This will generate the bundle in a subdirectory called dist
```

## How to run
First, run the main code below by

```bash
# Run the execution file by typing in
python main.py
```

This will show you an image like below,
[TODO] image

Drag an area you want to label. A green bounding box will appear as you drag along the area.
After you release your click, a window will automatically pop up like the below.
[TODO] label enter window

Type in the type and subtype of the given area.

If you want to set the area again, just press the 'Quit' button.

If you want to redo the entire labelling process, press 'r' in your keyboard.

If you want to stop the labelling, press 'q'.
