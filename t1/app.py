#coding=utf-8
from tkinter import *
import tkinter as tk
from example_words_prediction import Autocompleter
from combined_competer import CombinedAutocompleter
from PIL import Image, ImageTk


class Block:
    def __init__(self, master):
        self.master = master
        self.e = Entry(master, bg='white', fg='#696969', width=100, relief='groove', font='Times 30')
        self.b = Button(master, text="Complete", bg='Gainsboro', fg='black', font='Times 15')
        self.b1 = Button(master, text='', bg='#A9A9A9', fg='white', font='Times 25')
        self.b2 = Button(master, text='', bg='#A9A9A9', fg='white', font='Times 25')
        self.b3 = Button(master, text='', bg='#A9A9A9', fg='white', font='Times 25')
        self.b_clean = Button(master, text='Clean', bg='Gainsboro', fg='black', font='Times 15')
        self.e.pack()
        self.b.pack()
        self.b1.pack()
        self.b2.pack()
        self.b3.pack()
        self.b_clean.pack()
        # self.completer = Autocompleter()
        self.completer = CombinedAutocompleter()
        self.text = ''

        photo = tk.PhotoImage(file='D:\\Typing\\data\\friends.gif')
        self.label = Label(image=photo, relief='flat')
        self.label.image = photo
        self.label.pack(side='bottom')

    def setFunc(self, func):
        self.b['command'] = eval('self.' + func)

    def setFuncBut1(self):
        self.b1['command'] = eval('self.set_text1')

    def setFuncBut2(self):
        self.b2['command'] = eval('self.set_text2')

    def setFuncBut3(self):
        self.b3['command'] = eval('self.set_text3')

    def setFuncButClean(self, func):
        self.b_clean['command'] = eval('self.'+func)

    def complete(self):
        text = self.e.get()
        completions = self.completer.complete(text)
        self.b1['text'] = completions[0]
        self.b2['text'] = completions[1]
        self.b3['text'] = completions[2]

    def set_text1(self):
        text = self.b1['text']
        ending = len(self.e.get().split(' ')[-1])
        self.e.insert(len(self.e.get()), text[ending:]+' ')
        self.complete()

    def set_text2(self):
        text = self.b2['text']
        ending = len(self.e.get().split(' ')[-1])
        self.e.insert(len(self.e.get()), text[ending:]+' ')
        self.complete()

    def set_text3(self):
        text = self.b3['text']
        ending = len(self.e.get().split(' ')[-1])
        self.e.insert(len(self.e.get()), text[ending:]+' ')
        self.complete()

    def clean(self):
        self.e.delete(0, len(self.e.get()))
        self.b1['text'] = ''
        self.b2['text'] = ''
        self.b3['text'] = ''


root = Tk()
root.geometry('800x600')
root.title('Autocomplete')
root["bg"] = "white"
first_block = Block(root)
first_block.setFunc('complete')
first_block.setFuncBut1()
first_block.setFuncBut2()
first_block.setFuncBut3()
first_block.setFuncButClean('clean')

# second_block = Block(root)
# second_block.setFunc('strReverse')

root.mainloop()