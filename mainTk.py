# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:42:41 2020

@author: charl
"""

import tkinter as tk
#from tkinter import * 


#creation de l'espace de saisie
class menu():
    def __init__(self,main):
        #main frame
        self.main=main
        self.bg=tk.Frame(self.main, bg="red")
        self.bg.grid(row=0, column=1, sticky="nsew")
        
        #entry 
        for i in range(4):
           self.bg.columnconfigure(i, weight=1)
           
        self.bg.columnconfigure(0, weight=1)
        
        #check/reset
        self.check=tk.Button(self.bg)
        self.check.grid(row=3, column=0, sticky="nsew")

#création de l'espace de dessin 
class net():
    def __init__(self,main):
        self.main=main
        self.bg=tk.Frame(self.main, bg="green")
        self.bg.grid(row=0, column=0, sticky="nsew")
        
        




if __name__ == "__main__":
#fenetre principale
    main = tk.Tk()
    main.title('Réseau bayésian')
    main.geometry("1300x700+300+100")
    main.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=3)
    main.columnconfigure(1, weight=1)
    
    
#menu=saisie des noeuds et arc du réseau
    set_menu=menu(main)
    
#net = canva pour la visualisation des noeud et la saisie des domaines et probas
    #set_net=net(main)
    
#afficher 
    main.mainloop()





