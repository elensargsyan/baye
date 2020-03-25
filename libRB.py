#!/usr/bin/python3
# -*- coding: utf-8 -*-
#

"""
B{project I{libRB}}: I{classes}:
    - sommet,
    - graphe,
    - reseauBayesien

    L'objectif est de faire de l'inference bayesienne exacte
    en utilisant le passage de messages, les calculs se font
    de maniere locale. Ne marche que dans le cadre des DAGs.
"""
__author__ = "mmc <marc-michel dot corsini at u-bordeaux dot fr>"
__version__ = "3.02"
__date__ = "16.03.16"
__usage__ = "ressources pour le RB"
__revision__ = "08.01.17"
#---- import -------------------------------------
import itertools
import numpy as np # gestion des matrices de proba
#---- sommet -------------------------------------
class sommet(object):
    """
    gestion des sommets du graphe
        >>> x = sommet('u')
        creation du sommet de nom "u"
    """
    _index = 1
    def __init__(self,nom):
        """
        initialisation
        @param nom: une chaine de caracteres
        """
        assert isinstance(nom,str)
        self.__name = nom
        self.__index = sommet._index
        sommet._index += 1
        self.__pred = []
        self.__succ = []
        self.__type = None
        self.__degp, self.__degm = 0,0
        # les variables suivantes sont liees au RB
        self.__dom = [] # domaine Omega_X
        self.__unary = None # flag
        self.__proba = None # table proba du sommet
        # emissions / receptions
        self.__msgRecus = {}
        self.__valeur = None # la valeur que prend le noeud affecte
        self.__oneCond = {} # table proba conditionnelle indexee sur pred
        self.__diagM = None # Pr(X|Ev+)
        self.__causeV = None # Pr(E-|X)

    #----- variables temporaires pour mise au point --------
    @property
    def matrice(self):
        """
        permet de visualiser la matrice construite par
        __combineCausality
        """
        return self.__diagM
    @property
    def vecteur(self):
        """
        permet de visualiser le vecteur construit par
        __combineDiagnostic
        """
        return self.__causeV
    @property
    def messages(self):
        """
        Permet de visualiser l'ensemble des messsages recus par le sommet
        """
        return self.__msgRecus
    #-------------------------------------------------------
    
    def __str__(self):
        """ permet un affichage du sommet """
        _s = ''
        _s += 'nom: %s(%d)\n' % (self.nom,self.idx)
        _s += 'genre: %s\n' % self.genre
        _s += 'pred: %s\n' % self.pred
        _s += 'succ: %s\n' % self.succ
        _s += 'dom: %s\n' % str(self.domaine)
        _s += 'proba: %s\n' % self.proba
        _s += 'distribution: %s' % self.valeur
        if self.proba is not None and len(self.messages) == self.degre :
            _s += '\nestimation: %s' % self.croyance
        return _s
    
    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__,self.nom)
    
    @property
    def idx(self):
        """ numero du sommet """
        return self.__index
    
    @property
    def domaine(self):
        """ domaine des valeurs (Omega) """
        return tuple(self.__dom)
    @domaine.setter
    def domaine(self,v):
        """ pas de doublons dans Omega """
        print("dom of %s is" % self.nom, v)
        assert isinstance(v,(list,tuple))
        for x in v :
            if x not in self.domaine :
                self.__dom.append(x)

    def resetDomaine(self):
        """ efface le domaine et les tables de probas """
        self.__dom = []
        self.__proba = None
    
    @property
    def valeur(self):
        """ une distribution des valeurs du domaine """
        return self.__valeur
    @valeur.setter
    def valeur(self,v):
        """
        affecte la connaissance du sommet
        ne devrait pas etre utilisee ailleurs que pour la
        creation des observations dans le RB
        """
        assert v in self.domaine or len(v)==len(self.domaine)
        if v in self.domaine :
            self.__valeur = np.zeros( len(self.domaine), float )
            self.__valeur[self.domaine.index(v)] = 1.
        else:
            self.__valeur = np.array( v , float) # on cree les valeurs
            self.__valeur /= np.sum(self.__valeur) # norm

    def resetValeur(self):
        """
        reinitialise la connaissance sur le noeud
        """
        self.__valeur = None
        self.__msgRecus = {}
        self.__causeV = None
        self.__diagM = None
        for x in self.succ:
            self.__msgRecus[x] = np.ones(len(self.domaine))
            
    @property
    def proba(self):
        """ renvoie la table de probabilite associee au noeud """
        if self.__proba is not None :
            return self.__normaliseProbability()
    
    
    def __get_pred(self):
        """ liste des predecesseurs directs """
        return self.__pred
    
    def __set_pred(self,v):
        """
        affecte les predecesseurs du sommet
        met a jour les differentes tables
        ne devrait pas etre utilise ailleurs que la construction du graphe
        """
        assert isinstance(v,str)
        assert v not in self.pred
        #assert v not in self.succ : DAG only
        self.__pred.append(v)
        self.__degm += 1
        
    def __del_pred(self):
        self.__pred = []
        self.__degm = 0
        
    pred = property(__get_pred,__set_pred,__del_pred)
    
    
    def __get_succ(self):
        """ liste des successeurs directs """
        return self.__succ
    
    def __set_succ(self,v):
        """
        affecte les successeurs du sommet
        met a jour les differentes tables
        ne devrait pas etre utilise ailleurs que la construction du graphe
        """
        assert isinstance(v,str)
        #assert v not in self.pred : DAG only
        assert v not in self.succ
        self.__succ.append(v)
        self.__degp += 1
        
    def __del_succ(self):
        self.__succ =[]
        self.__degp = 0
        
    succ = property(__get_succ,__set_succ,__del_succ)

    @property
    def degre(self):
        """ visualise le degre du sommet """
        return self.__degm + self.__degp
    @property
    def degIN(self):
        """ visualise le degre entrant du sommet """
        return self.__degm
    @property
    def degOUT(self):
        """ visualise le degre sortant du sommet """
        return self.__degp
    @property
    def genre(self):
        """ visualise la nature du sommet root, leaf, internal """
        if self.__type is not None :
            assert self.__type in ('root','leaf','internal')
            if self.__type == 'root' : assert self.pred == []
            elif self.__type == 'leaf' : assert self.succ == []
            else: assert self.pred != [] and self.succ != []
        return self.__type
    @genre.setter
    def genre(self,v):
        """
        affecte le genre du sommet apres controle
        ne devrait pas etre utilise ailleurs que setup
        """
        assert v in ('root','leaf','internal')
        if v == 'root' : assert self.pred == []
        elif v == 'leaf' : assert self.succ == []
        else: assert self.pred != [] and self.succ != []
        self.__type = v

    @property 
    def nom(self):
        """ visualise le nom du sommet """
        return self.__name
                
    def setup(self):
        """ calcul le genre du sommet """
        if self.pred == [] : self.genre = 'root'
        elif self.succ == [] : self.genre = 'leaf'
        else: self.genre = 'internal'

    # gestion des probabilites
    def __probaOneCond(self,sommet):
        """
        I{methode B{privee}}:
        passe de p(X | Z1 ... Zn) a p(X | Zi)
        @param sommet: le sommet conditionnant (Zi)
            - les axes sont 0 pour X, 1 pour Z1 ...
            - self.pred ne contient pas X, il faut decaler
            - si pred = [ .. ] et que l'on veut recuperer Zi
                - len(pred) = n
                - 0 <= pred.index(Zi) < n : position de Zi
                - l'axe de Zi est donc pred.index(Zi) + 1
        @return: Pr(self|vertex)
        """
        assert sommet in self.pred, "%s n'est pas un ancetre de %s" %\
               (sommet,self.nom)
        _nperes = len(self.pred)
        _mat = self.proba
        j = self.pred.index(sommet) # index a preserver
        for i in range(_nperes):
            if i < j : _mat = np.apply_along_axis(sum,1,_mat)
            elif i == j : continue
            else: _mat = np.apply_along_axis(sum,2,_mat)
        return _mat
    
    def getProbSachant(self,vertex):
        """
        calcul de Pr(X|Zi) a partir de la table Pr(X|Z1 ... Zn)
        
        @param vertex: le sommet conditionnant
        @return:
        une matrice Omega_self x Omega_vertex aka P(self|vertex)
        """
        assert vertex in self.pred, "%s pas predecesseur de %s" %\
               (vertex,self.nom)
        if self.__unary : return self.__oneCond[vertex]

    @property
    def belief(self):
        """
        visualise la croyance d'un sommet
        par definition c'est : Pr(X|Ev)
        """
        _combine = self.causality * self.diagnostic
        if self.valeur is not None: return self.valeur 
        return _combine
            
    def normalizedBelief(self):
        """
        met belief sous forme de vecteur probabilite (normalisation)
        @return: la distribution normalisee
        """
        _dist = self.belief
        return _dist / np.sum(_dist)
    croyance = property(normalizedBelief)
    #helper
    def __combineCausality(self):
        """
        I{methode B{privee}}:
        calcul de Pr(X|E1+, ..., En+)
        @return: Pr(X|Ev+)
        """
        if self.__diagM is not None : return self.__diagM
        if self.pred == []: # je suis racine
            if self.valeur is not None : return self.valeur
            else: return self.proba
        _base = self.getMsgFrom(self.pred[0])
        _nperes = len(self.pred)
        for i in range(1,_nperes):
            _base = np.multiply.outer(_base,
                                      self.getMsgFrom(self.pred[i]))
        self.__diagM = np.tensordot(self.proba,_base,_nperes)
        return self.__diagM
    causality = property(__combineCausality)

    def __combineCausalityExcept(self,pere):
        """
        I{methode B{privee}}:
        calcul de Pr(X|E1+, ..., E(i-1)+,E(i+1)+, .., En+)
        @return: causality for others
        """
        if self.pred == []: # racine
            if self.valeur is not None : return self.valeur
            else: return self.proba
        _nperes = len(self.pred)
        _i = self.pred.index(pere)
        _res = self.proba
        for j in range(_nperes):
            _msg = self.getMsgFrom(self.pred[j])
            if j < _i : _res = np.tensordot(_res,_msg,(1,0))
            elif _i == j : continue
            else:
                _res = np.tensordot(_res,_msg,(2,0))
        return _res
    
    #helper
    def __combineDiagnostic(self):
        """
        I{methode B{privee}}:
        calcul de Pr(E1-, ..., Em- | X)
        le calcul de __causeV se fait incrementalement
        a chaque reception d'un diagnostic

        @return: Pr(Ev-|X)
        """
        if self.__causeV is None :
            if self.valeur is None :
                _base = np.ones( len(self.domaine) )
            else:
                _base = self.valeur
            for x in self.succ :
                _base *= self.getMsgFrom(x)
            self.__causeV = _base
        return self.__causeV
        
    diagnostic = property(__combineDiagnostic)

    def getMsgFrom(self,vertex):
        """
        Le message recu precedemment de vertex
        
        @param vertex: sommet emetteur
        @return: message connu
        """
        return self.__msgRecus.get(vertex,None)
    
    def getDiagnostic(self,fils,msg):
        """
        recoit le diagnostique de fils
        gere __msgRecus[fils]

        si le msg est different de ce que l'on sait
        il faut recalculer __msgEmis
        
        @param fils: le sommet emetteur
        @param msg: le support envoye par le fils
        """
        assert not np.all(msg == self.getMsgFrom(fils))
        assert self.__causeV is not None, 'hum!'
        self.__causeV /= self.getMsgFrom(fils) #on enleve l'ancienne cause
        self.__msgRecus[fils] = msg # maj msg
        self.__causeV *= msg # maj vecteur causes
    
    def sendDiagnostic(self,pere):
        """
        envoie a pere mon diagnostic pour lui
        @return: Pr(Ev-  | Zj)
        """
        _mat = self.getProbSachant(pere)
        if len(self.pred) != 1:
            _mat = self.__combineCausalityExcept(pere)
        _msg = np.dot(self.diagnostic,_mat)
        return _msg #/ np.sum(_msg) # normalisation

    def getCausality(self,pere,msg):
        """
        recoit la causalite de mon parent
        gere __msgRecus[pere]

        si le msg est different de ce que l'on sait
        il faut recalculer __msgEmis

        @param pere: le sommet emetteur
        @param msg: le support envoye par le pere
        """
        _oldMsg = self.getMsgFrom(pere)
        if _oldMsg is not None: assert not np.all(msg == _oldMsg)
        self.__diagM = None # refaire matrice
        self.__msgRecus[pere] = msg
    
    def sendCausality(self,fils):
        """
        envoie a fils ma causalite pour lui
        @return: Pr(X | Ev+,E1-,..E(i-1)-,E(i+1)-, .. En-)
        """
        _msg = self.belief / self.getMsgFrom(fils)
        return _msg #/ np.sum(_msg) # normalisation
    
    def initProbability(self,dim):
        """
        creation du numpy.array
        """
        self.__proba = np.zeros( dim, float )
        self.__unary = False
        
    def setProbability(self,index,val):
        """
        Permet de saisir les probabilites associees au noeud
        @param index: un n-uplet
        @param val: une valeur numerique
        """
        self.__proba[index] = val
        self.__unary = False


    def buildProbabilityTable(self):
        """
        Permet de construire l'ensemble des p(X|Z)
        DOIT etre appele apres avoir fait la mise a jour
        des probabilites conditionnelles
        """
        for x in self.pred :
            self.__oneCond[x] = self.__probaOneCond(x)
        self.__unary = True
        
    # helper
    def __normaliseProbability(self):
        """
        I{methode B{privee}}:
        Met la table de probabilite au format
        """
        if not self.__unary :
            self.__proba /= np.apply_along_axis(sum,0,self.__proba)
            self.__unary = True
        return self.__proba
#---- graphe ------------------------------------
class graphe(object):
    """
    gestion du graphe
        >>> g = graphe(['o','l','c'],[('o','l'),('o','c')])
        creation d'un graphe g possedant 3 sommets "o", "l" et "c"
        et 2 arcs "o" -> "l" ; "o" -> "c"
    """
    def __init__(self,vertices=[],edges=[]):
        """
        constructeur du graphe
        @param vertices: liste des noms des sommets
        @param edges: liste des arcs du graphe
        """
        assert isinstance(vertices,list)
        assert isinstance(edges,list)
        self.__dicNodes = {}
        self.__dicLinks = {}
        self.__root = set([])
        self.__leaves = set([])
        self.__internal = set([])
        for x in vertices : self.addNode(x)
        for x,y in edges : self.addLink(x,y)
        # flags
        self.modification = True
        self.update()
       
    def __str__(self):
        """ mise en caracteres du graphe """
        _s = ''
        for x in self.noeuds :
            _s += str(self.noeud(x))+'\n'+('-'*5)+'\n'
        return _s
        
    @property
    def modification(self): return self.__modified
    @modification.setter
    def modification(self,v):
        """ on se moque de v """
        self.__modified = True
        self.__ordered = None
        self.__dicfc = None
        self.__connex = None
        
    @property
    def noeuds(self):
        """
        visualise les noeuds du graphe
        @return: liste des noms des sommets du graphe

            >>> g.noeuds
            [ 'o', 'l', 'c' ]

        la sortie est indexee par rapport aux idx des noeuds
        """
        return sorted(self.__dicNodes.keys(),
                      key=lambda _: self.__dicNodes[_].idx)
    @property
    def arcs(self):
        """
        visualise les arcs du graphe
        @return: liste des noms des arcs du graphe

            >>> g.arcs
            [ ('o','l'), ('o','c') ]

        l'ordre des arcs saisis influe sur la saisie des probas
        """
        return self.__dicLinks.keys()

    @property
    def nbSommets(self):
        """
        visualise le nb sommets du graphes

            >>> g.nbSommets
            3
        """
        return len(self.__dicNodes)
    @property
    def nbArcs(self):
        """
        visualise le nb arcs du graphe

            >>> g.nbArcs
            2
        """
        return len(self.__dicLinks)
    @property
    def racine(self):
        """ visualise la liste des sommets racine du graphe """
        return self.__root
    @property
    def feuille(self):
        """ visualise la liste des feuilles du graphe """
        return self.__leaves
    @property
    def interne(self):
        """ visualise la liste des sommets internes du graphe """
        return self.__internal

    def noeud(self,x):
        """
        accesseur sur sommet(x)
        
        @param x: nom du sommet
        @type x: chaine de caracteres
        @return: sommet(x)
        """
        return self.__dicNodes[x]
        
    def link(self,e):
        """
        accesseur sur (sommet(x), sommet(y))
        
        @param e: un arc
        @type e: un tuple (sommet source, sommet but)
        @return: un tuple (sommet(e[0]), sommet(e[1]))
        """
        return self.__dicLinks[e]
    
    def addNode(self,x):
        """ ajoute un noeud x dans le graphe """
        assert x not in self.noeuds
        self.modification = True     
        node = sommet(x)
        self.__dicNodes[x] = node
        
    def addLink(self,x,y):
        """ ajoute un arc entre deux sommets x -> y """
        assert x in self.noeuds
        assert y in self.noeuds
        if x == y : print("loops are forbidden : {} -> {} rejected".format(x,y)) ; return
        assert (x,y) not in self.arcs
        self.modification = True # graphe modifie
        self.__dicLinks[ (x,y) ] = 0
        self.__dicNodes[x].succ = self.noeud(y).nom
        self.__dicNodes[y].pred = self.noeud(x).nom

    def update(self):
        """
        Mise a jour des genres des sommets du graphe
        appel setup pour chaque sommet
        """
        self.modification = True
        for x in self.noeuds:
            self.noeud(x).setup()
            if self.noeud(x).genre == 'root' :
                self.__root.add(x)
            elif self.noeud(x).genre == 'leaf' :
                self.__leaves.add(x)
            else: self.__internal.add(x)
        self.__modified = False
        
    @property
    def topological(self):
        """
        fait un tri topologique sur les sommets
        @return: la liste des sommets tries
        (pred < sommet < succ)
        """
        if self.__ordered : return self.__ordered
        if not self.isDAG: return []
        self.update()
        # initialisation
        for x in self.noeuds : 
            self.noeud(x).couleur = 0
            for attr in "debut fin pi".split(): setattr(self.noeud(x),attr,None)
        for a in self.arcs : self.__dicLinks[a] = 0
        _order = [] # le tri topologique
        self.__date = 0
        # le parcours en profondeur
        for x in self.racine :
            if self.noeud(x).couleur == 0 :
                _order = self.__visitPP(x,_order)
        self.__ordered = _order
        return self.__ordered
    
    # helper
    def __visitPP(self,x,ordre):
        """
        I{methode B{privee}}:
        remplit ordre apres avoir visiter tous les fils
        """
        self.noeud(x).couleur = 1
        self.noeud(x).debut = self.__date
        self.__date += 1
        for y in self.noeud(x).succ :
            self.__dicLinks[ (x,y) ] = self.noeud(y).couleur
            if self.noeud(y).couleur == 0 :
                self.noeud(y).pi = x
                ordre = self.__visitPP(y,ordre)
        self.noeud(x).couleur = 2
        self.noeud(x).fin = self.__date
        self.__date += 1
        ordre.insert(0,x)
        return ordre

    def cfc(self,verbose=False):
        """ calcul les composantes fortement connexes du graphe """
        if self.__dicfc : return self.__dicfc
        for x in self.noeuds : 
            self.noeud(x).couleur = 0
            for attr in "debut fin pi".split(): setattr(self.noeud(x),attr,None)
        # le parcours en profondeur
        self.__date = 0 ; _order = []
        for x in self.noeuds :
            if self.noeud(x).couleur == 0 :
                _order = self.__visitPP(x,_order)
        if verbose:
            for x in self.noeuds: print("{0} [{1.debut} | {1.fin}] parent {1.pi}".format(x,self.noeud(x)))
        dates = sorted(self.noeuds,key=lambda _:self.noeud(_).fin,reverse=True)
        for x in self.noeuds :
            self.noeud(x).couleur = 0
            for attr in "debut fin pi".split(): setattr(self.noeud(x),attr,None)
        # le parcours en profondeur dans le graphe transposé
        self.__date = 0
        dicfc = {}
        for x in dates:
            _order = []
            if self.noeud(x).couleur == 0 :
                _order = self.__visitRP(x,_order)  
                if verbose: print("%3s >" % x,_order)
                dicfc[x] = _order
        self.__dicfc = dicfc
        return dicfc
        
    # helper
    def __visitRP(self,x,ordre):
        """
        I{methode B{privee}}:
        remplit ordre apres avoir visiter tous les fils
        """
        self.noeud(x).couleur = 1
        self.noeud(x).debut = self.__date
        self.__date += 1
        for y in self.noeud(x).pred :
            self.__dicLinks[ (x,y) ] = self.noeud(y).couleur
            if self.noeud(y).couleur == 0 :
                self.noeud(y).pi = x
                ordre = self.__visitRP(y,ordre)
        self.noeud(x).couleur = 2
        self.noeud(x).fin = self.__date
        self.__date += 1
        ordre.insert(0,x)
        return ordre     
        
    @property
    def hasCircuit(self):
        """ renvoie vrai si présence de circuit - pas de boucle par construction du graphe """
        _out = self.cfc()
        if len(_out) != self.nbSommets : return True
        return False
        
    @property
    def isDAG(self):
        """
        Renvoie Vrai si le graphe est un DAG, faux sinon
        """
        if not self.isConnex : return False
        if self.hasCircuit : return False
        return True
        # # on fait un tri topologique
        # _ = self.topological
        # if len(set(_)) != self.nbSommets : return False
        # # on regarde la couleur des arcs
        # _ok = [ self.link(a) for a in self.arcs ]
        # # print(_ok)
        # return _ok.count(1) == 0

    @property
    def isConnex(self):
        """
        Renvoie vrai si le graphe est connexe
        """
        if self.__connex is not None : return self.__connex
        if self.noeuds == []: return False
        for x in self.noeuds: self.noeud(x).seen = False
        _toDO = [self.noeuds[0]]
        while _toDO != [] :
            x = _toDO.pop()
            if not self.noeud(x).seen :
                _toDO.extend( self.noeud(x).succ )
                _toDO.extend( self.noeud(x).pred )
                self.noeud(x).seen = True
        self.__connex = all([self.noeud(x).seen for x in self.noeuds])
        for x in self.noeuds: del self.noeud(x).seen
        self.__modified = False
        return self.__connex
        
    def descendants(self,vertex,strict=True):
        """
        calcul l'ensemble des descendants d'un sommet
        @param vertex: le sommet de depart
        @return: la liste de tous les descendants
        """
        if strict: _todo = self.noeud(vertex).succ
        else: _todo = [vertex]
        _done = []
        while _todo != [] :
            _next = _todo.pop(0)
            if _next not in _done :
                _done.append(_next)
                _todo.extend(self.noeud(_next).succ)
        return _done
    
    def ancetres(self,vertex,strict=True):
        """
        calcul l'ensemble des parents d'un sommet
        @param vertex: le sommet de depart
        @return: la liste de tous les descendants
        """
        if strict: _todo = self.noeud(vertex).pred
        else: _todo = [vertex]
        _done = []
        while _todo != [] :
            _next = _todo.pop(0)
            if _next not in _done :
                _done.append(_next)
                _todo.extend(self.noeud(_next).pred)
        return _done

    def __eq__(self,other):
        """ première version non satisfaisante """
        if isinstance(other,self.__class__):
            return str(self) == str(other)
        return False
        
class reseauBayesien(object):
    """
    Outils pour la manipulation des RB
        >>> g = graphe(['o','l','c'],[('o','l'),('o','c')])
        creation du graphe g
        >>> grb = reseauBayesien(g)
        creation du reseau grb base sur le graphe g
    """
    def __init__(self,graph):
        """
        constructeur du reseau
        
        @param graph: le graphe du reseau
        """
        assert isinstance(graph,graphe)
        self.__gr = graph
        self.__initialized = False

    def __str__(self):
        """ affichage du graphe sous-jacent """
        return str(self.graphe)

    @property
    def graphe(self):
        """ le graphe sous-jacent """
        return self.__gr
    
    def test(self):
        """
        un jeu de tests pour ordonner les sommets
        """
        _din, _dout = 0,0
        for x in self.graphe.noeuds :
            _din += self.graphe.noeud(x).degIN
            _dout += self.graphe.noeud(x).degOUT
        _msgv = "degre entrant %d = degre sortant %d = nombre d'arcs %d\n" %\
              (_din,_dout,self.graphe.nbArcs)
        _msgv += "tri topologique, un sommet avant ses successeurs : \n"
        _rep = self.graphe.topological
        assert _rep[0] in self.graphe.racine, "erreur de calcul"
        print (_msgv,_rep)
        _count = 0
        _arcs = []
        for a in self.graphe.arcs :
            #pour controle 
            #print self.graphe.link(a),
            if self.graphe.link(a) == 1 :
                _count += 1
                _arcs.append(a)
        if _count != 0 :
            _msgv = '\ndetection de circuit passant par %s' % _arcs
        else:
            _msgv = 'On a un DAG'
        print( _msgv )
        
    def setDomaine(self,vertex,dom):
        """
        affecte un domaine dom au sommet vertex
            >>> grb.setDomaine('o',['x','y'])
            le sommet "o" a un domaine de taille 2
        """
        assert vertex in self.graphe.noeuds # c'est un noeud
        self.graphe.noeud(vertex).resetDomaine() # raz domaine
        self.graphe.noeud(vertex).domaine = dom
        
    def setDomaines(self):
        """
        affecte les domaines des variables du reseau
        methode fastidieuse
        """
        for x in self.graphe.noeuds :
            print("domaine de %s ? " % x, end=" ")
            self.graphe.noeud(x).domaine = input().split()

    def resetDomaines(self):
        """ remet a vide les domaines des variables du reseau """
        for x in self.graphe.noeuds :
            self.graphe.noeud(x).resetDomaine()
        self.__initialized = False

    def enumDomaines(self,vertex):
        """
        iterateur sur les valeurs du domaine de vertex

        @param vertex: 
            le sommet dont on veut calculer les probas
        @return: un generateur sur les domaines
             
        - dom(x) = ['a', 'b'] 
        - dom(y) = [ 1,2,3 ] 
        - dom(z) = [7,8]
            - z -> x, y -> x ; 
            on veut gerer p(x | y,z) il y a 12 combinaisons

        l'iterateur produira (a,1,7) (a,1,8) (a,2,7) ... (b,3,8)
        """
        _support = [ self.graphe.noeud(vertex).domaine ]
        for y in self.graphe.noeud(vertex).pred :
            _support.append( self.graphe.noeud(y).domaine )

        return itertools.product(* tuple(_support)) 

    def enumIndex(self,vertex):
        """
        iterateur sur les index des valeurs du domaine de vertex
        
        @param vertex: 
            le sommet dont on veut calculer les probas
        @return: un generateur sur les domaines
             
        - dom(x) = ['a', 'b'] 
        - dom(y) = [ 1,2,3 ] 
        - dom(z) = [7,8]
            - z -> x, y -> x ; 

        on veut gerer p(x | y,z) il y a 12 combinaisons
        l'iterateur produira (0,0,0) (0,0,1) ... (1,2,1)
        """
        _support = [ range(len(self.graphe.noeud(vertex).domaine)) ]
        for y in self.graphe.noeud(vertex).pred :
            _support.append( range(len(self.graphe.noeud(y).domaine)) )

        return itertools.product(* tuple(_support)) 

    def resetProbability(self):
        """
        initialise les tables de probabilites pour chaque noeud
        On parcourt les noeuds du reseau,
        On recupere les dimensions des domaines
        On initialise la table du noeud
        """
        for x in self.graphe.noeuds :
            # pour chaque noeud on cree le n-uplet des dimensions
            _dim = [ len(self.graphe.noeud(x).domaine) ]
            for y in self.graphe.noeud(x).pred :
                _dim.append( len(self.graphe.noeud(y).domaine) )
            self.graphe.noeud(x).initProbability( _dim )

        self.__initialized = False

    def setProbability(self):
        """
        initialisation des tables de probabilites du reseau
        methode fastidieuse ou on rentre l'info une par une.

        Fait appel a resetProbability et aux iterateurs
        """
        # initialisation des tables de probas
        self.resetProbability()
        # affectation des tables de probabilites
        for x in self.graphe.noeuds : #chaque noeud
            for nom,idx in zip(self.enumDomaines(x),
                               self.enumIndex(x) ) :
                _v = float(input( "proba de %s=%s sachant %s : " %\
                                      (x,nom[0],nom[1:]) ))
                self.graphe.noeud(x).setProbability(idx,_v)
            # la table est construite pour x
            self.graphe.noeud(x).buildProbabilityTable()
            
    def setEvidence(self,dicEv={}):
        """
        affecte la connaissance de la distribution d'un sommet
        
        @param dicEv: un dictionnaire
        @type dicEv: { key: val, .. }
        @return: liste des clefs valides
        
        >>> grb.setDomaine('o',['x','y'])
        >>> grb.setDomaine('c',('a','aa','aaa'))
        >>> grb.setDomaine('l',range(2))
        Affectation des domaines
        >>> grb.setEvidence( { 'o' : 'x' , 'c' : (1 , 2 , 0 ) , 'l': 1 } )
        Pr(o = x) = 1 ; Pr(o) = (1 0)
        Pr(c) = (1/3 2/3 0)
        Pr(l = 1) = 1 ; Pr(l) = (0 1)
        """
        assert isinstance(dicEv,dict)
        if dicEv == {}: return {}
        _k = dicEv.keys()
        _nodes = self.graphe.noeuds
        _validk = [ x for x in _k if x in _nodes ]
        _dico = {}
        for x in _validk : # on met les valeurs a jour si possible
            try:
                self.graphe.noeud(x).valeur = dicEv[x]
                _dico[x] = dicEv[x]
            except Exception as _e:
                print( _e )
        return _dico
    
    def doUp(self,frm,to,verbose=False):
        """
        effectue une propagation ascendante du support diagnostic
        to est un noeud serie pour from vers pred(to)
        to est un noeud divergent pour from vers succ(to)
        il suffit que to soit instancie pour bloque le message
        
        @param frm: sommet fils
        @param to: sommet pere
        @param verbose: [B{defaut}: False] permet d'avoir des messages
        pour tracer les calculs
        @return: None
        """
        assert to in self.graphe.noeud(frm).pred,\
               "%s n'est pas pere de %s" % (to, frm)
        _sender = self.graphe.noeud(frm)
        _reciever = self.graphe.noeud(to)
        _msg = _sender.sendDiagnostic(to)
        if verbose:
            _msgv = '%s |-> %s : new %s old %s' %\
                  (frm,to,_msg,_reciever.getMsgFrom(frm))
            print(_msgv)
        if (_reciever.valeur is None and
            not np.allclose(_msg,_reciever.getMsgFrom(frm),equal_nan=True)):
            # mise a jour
            if verbose:
                print ('modification du diagnostic de %s' % to)
            _reciever.getDiagnostic(frm,_msg)
            # reexpedition a tous les parents de to
            for x in _reciever.pred:
                self.doUp(to,x,verbose)
            # reexpedition a tous les fils de to sauf frm
            for x in _reciever.succ:
                if x != frm :
                    self.doDown(to,x,verbose)

    def doDown(self,frm,to,verbose=False):
        """
        effectue une propagation descendante du support causal
        to est un noeud serie de frm vers succ(to)
        si to est instancie : il bloque
        to est un noeud convergent de frm vers pred(to)
        si to ou un succ(to) est instancie : il passe
        
        @param frm: sommet pere
        @param to: sommet fils
        @param verbose: [B{defaut}: False] permet d'avoir des messages
        pour tracer les calculs
        @return: None
        """
        assert to in self.graphe.noeud(frm).succ,\
               "%s n'est pas fils de %s" % (to,frm)
        _sender = self.graphe.noeud(frm)
        _reciever = self.graphe.noeud(to)
        _msg = _sender.sendCausality(to)
        if verbose:
            _msgv = '%s -> %s : new %s old %s' %\
                (frm,to,_msg,_reciever.getMsgFrom(frm))
            print( _msgv )
        # on regarde si le msg a change
        _oldMsg = _reciever.getMsgFrom(frm)
        if _oldMsg is None: _same = False
        else: _same = np.allclose(_msg,_oldMsg,equal_nan=True)
        # on regarde combien a d'ancetres le sommet
        _nperes = _reciever.degIN
        # on regarde si c'est un noeud observe
        _ev = (_reciever.valeur is not None)
        # on applique les regles de propagations
        if _same:
            if verbose:
                print( 'pas de modification arret' )
            return # blocage
        if (_ev and _nperes == 1):
            if verbose:
                print( '%s connu, dans un arbre arret' % to )
            return # blocage
        # mise a jour
        if verbose:
            print( 'modification des causes de %s ...' % to, end = " ")
        _reciever.getCausality(frm,_msg)
        if verbose:
            print( ' maj de %s effective' % to )
        # si plusieurs parents on regarde si un descendant affecte
        _ev2 = _ev
        # descendants
        
        if _nperes > 1 and not _ev2 :
            # recherche des descendants observes
            _svt = self.graphe.descendants(to,strict=False)
            assert _svt[0] == to, 'bizarre'
            _svt.pop(0) # on connait le premier
            while _svt != [] and not _ev2 :
                _fils = _svt.pop(0)
                _ev2 = (self.graphe.noeud(_fils).valeur is not None)
            if verbose:
                if _ev2 :
                    _msgv = '%s observe, %s' %\
                        (_fils,self.graphe.noeud(_fils).valeur)
                else: _msgv = 'aucune observation'
                print( _msgv )
        if _ev2 : 
            # reexpedition a tous les parents de to sauf frm
            if verbose and _nperes > 1 :
                print( '%s connu et plusieurs parents' % to )
            for x in _reciever.pred: # rebond
                if x != frm:
                    self.doUp(to,x,verbose)
        if (not _ev) :
            # on s'assure que tous les parents ont deja emis une fois
            _ok = True
            for x in _reciever.pred:
                _ok = _ok and (_reciever.getMsgFrom(x) is not None)
            if not _ok:
                if verbose:
                    print( 'propagation impossible depuis %s, arret' % to )
                return
            if verbose:
                print( '%s non observe on passe aux descendants' % to )
            # reexpedition a tous les fils de to 
            for x in _reciever.succ:
                if verbose:
                    print( '%s envoie vers %s' % (to,x) )
                self.doDown(to,x,verbose)
                if verbose:
                    print( '%s a fait son envoi vers %s' % (to,x) )
            if verbose:
                print( 'fin du traitement de %s' % to )

    def doInit(self,verbose=False):
        """
        effectue l'initialisation des calculs
        enleve les calculs prealables
        fait appel a doDown et doUp
        @param verbose: [B{defaut}: False] permet d'avoir des messages
        pour tracer les calculs
        @return: None
        """
        # phase 1 : reinitialisation des observations
        for x in self.graphe.noeuds:
            self.graphe.noeud(x).resetValeur()
        # phase 2 : propagation des informations
        for x in self.graphe.racine:
            for y in self.graphe.noeud(x).succ:
                self.doDown(x,y,verbose)
        self.__initialized = True
        
    def doPropage(self,evidence={},verbose=False):
        """
        Effectue la propagation des observations dans le reseau

        @param evidence: un dictionnaire
        @param verbose: [B{defaut}: False] permet d'avoir des messages
        pour tracer les calculs
        @return: None
        """
        if not self.__initialized:
            print("doInit has to be performed first")
            return None
        _dicEv = self.setEvidence(evidence)
        # _dicEv ce qui est reellement fait
        for x in _dicEv.keys():
            if self.graphe.noeud(x).pred == [] : # racine
                for y in self.graphe.noeud(x).succ:
                    self.doDown(x,y,verbose)
            elif self.graphe.noeud(x).succ == [] : # racine
                for y in self.graphe.noeud(x).pred:
                    self.doUp(x,y,verbose)
            else:
                raise TypeError( "operation non supportee pour %s" % x )
            
    def inferenceExacte(self,evidence={},verbose=False):
        """
        fait de l'inference exacte dans le reseau
        appel doInit doPropage et affiche le resultat
        @param evidence: [B{default}: {}] un dictionnaire
        @param verbose: [B{defaut}: False] permet d'avoir des messages
        pour tracer les calculs
        @return: None
        """
        self.doInit(verbose)
        if verbose: print( "*"*10 )
        self.doPropage(evidence,verbose)
        if verbose: print( "*"*10 )
        self.display(self.graphe.noeuds)

    def display(self,nodes):
        """
        affiche la croyance associee
        @param nodes: une serie de noeud
        """
        print( "Resultat de l'inference effectuee" )
        for x in nodes :
            print( "%10s %s" % (x,self.graphe.noeud(x).croyance) )
        print( '-'*5 )

def old_main():
    _sep = '='*10 
    g = graphe(['o','l','c'],[('o','l'),('o','c')])
    grb = reseauBayesien(g)
    grb.test()
    print( _sep )
    net = graphe(['h','e','l','f','p','b'],
                 [('h','e'),('h','l'),('b','l'),
                  ('e','f'),('l','p')])
    gnet = reseauBayesien(net)
    gnet.test()
    print( _sep )
    loop = graphe(['z','x','y'],[('z','x'),('x','y'),('y','x')])
    gloop = reseauBayesien(loop)
    gloop.test()
    print( _sep )
    cyclic = graphe( ['u','v','w','x','y','z'],
                     [ ('u','x'), ('u','v'), ('v','y'), ('w','y'), ('w','z'),
                       ('x','v'), ('y','x'), ('z','z') ])
    gcyc = reseauBayesien(cyclic)
    gcyc.test()
    print( _sep )
    # on entre les domaines sous la forme v1 v2 .. vn
    # exemple : on off
    x = 'o'
    print( "domaine de %s ? " % x, end=" ")
    g.noeud(x).domaine = input().split()
    print( _sep )
    print( g )
    print( _sep )
    # On rentre les domaines "a la main"
    grb.setDomaine('o',['x','y'])
    grb.setDomaine('c',('a','aa','aaa'))
    grb.setDomaine('l',list(range(2)))
    # On teste la mise en place des evidences
    # O = x ; C a 2 fois plus de chance d'etre aa que a ; L est 1
    grb.setEvidence( { 'o' : 'x' , 'c' : (1 , 2 , 0 ) , 'l': 1 } )
    # affichage
    for x in g.noeuds:
        print( x,g.noeud(x).valeur )
    print( _sep )
    # creation des tables de probas (pas de controle)
    # Il faut reflechir a qque chose de plus "souple"
    grb.setProbability()
    print( grb )
    print( _sep )
    # on doit construire les tables pour chaque sommet
    g.noeud('o').buildProbabilityTable()
    g.noeud('c').buildProbabilityTable()
    g.noeud('l').buildProbabilityTable()
    # essayons de recuperer une table inconnue
    try:
        g.noeud('o').getProbSachant('x')
    except AssertionError as _e :
        print( _e )

    print( _sep )

    for x in g.noeuds:
        print( 'vecteur: %s' % g.noeud(x).vecteur )
        print( 'matrice: %s' % g.noeud(x).matrice )
        print( 'message: %s' % g.noeud(x).messages )

    print( _sep )

    grb.doInit()
    for x in g.noeuds:
        print( 'vecteur: %s' % g.noeud(x).vecteur )
        print( 'matrice: %s' % g.noeud(x).matrice )
        print( 'message: %s' % g.noeud(x).messages )

    print( _sep )
    grb.inferenceExacte({'o': (1,0)})
    
    print( _sep )
    print( g.noeud('c').proba )
    g.noeud('c').setProbability(0,[2,3])
    print( g.noeud('c').proba )
    g.noeud('c').setProbability(0,[2,3])
    g.noeud('c').setProbability(1,[4,5])
    g.noeud('c').setProbability(2,[4,2])
    print( g.noeud('c').proba )

    
    """
    Quelques exemples de manipulations avec numpy
    http://math.mad.free.fr/depot/numpy/base.html
    >>> import numpy as np
    >>> p = np.array([.2 , .8])
    >>> q = np.array([.3, .5, .2])
    >>> r = np.array([1., 0., .5, .4, .6, 1., 0,1.,.5,.6,.4,0])
    >>> m = r.reshape(2,2,3)
    >>> p
    array([ 0.2,  0.8])
    >>> m[0]
    array([[ 1. ,  0. ,  0.5],
           [ 0.4,  0.6,  1. ]])
    >>> m[0][1]
    array([ 0.4,  0.6,  1. ])
    >>> m[0][1]*q
    array([ 0.12,  0.3 ,  0.2 ])
    >>> m[0][0]*q
    array([ 0.3,  0. ,  0.1])
    >>> m[0][0]*q *.2
    array([ 0.06,  0.  ,  0.02])
    >>> m[0][1]*q *.8
    array([ 0.096,  0.24 ,  0.16 ])
    >>> np.outer(p,q)
    array([[ 0.06,  0.1 ,  0.04],
           [ 0.24,  0.4 ,  0.16]])
    >>> m * np.outer(p,q)
    array([[[ 0.06 ,  0.   ,  0.02 ],
            [ 0.096,  0.24 ,  0.16 ]],

           [[ 0.   ,  0.1  ,  0.02 ],
            [ 0.144,  0.16 ,  0.   ]]])
    >>> np.apply_along_axis(sum,0,m * np.outer(p,q))
    array([[ 0.06,  0.1 ,  0.04],
           [ 0.24,  0.4 ,  0.16]])
    >>> np.apply_along_axis(sum,0,(m * np.outer(p,q)))
    array([[ 0.06,  0.1 ,  0.04],
           [ 0.24,  0.4 ,  0.16]])
    >>> np.apply_along_axis(sum,1,(m * np.outer(p,q)))
    array([[ 0.156,  0.24 ,  0.18 ],
           [ 0.144,  0.26 ,  0.02 ]])
    >>> (m * np.outer(p,q))
    array([[[ 0.06 ,  0.   ,  0.02 ],
            [ 0.096,  0.24 ,  0.16 ]],
           [[ 0.   ,  0.1  ,  0.02 ],
            [ 0.144,  0.16 ,  0.   ]]])
    >>> np.sum((m * np.outer(p,q))[0])
    0.57600000000000007
    >>> np.sum((m * np.outer(p,q))[1])
    0.42400000000000004
    >>> m.shape
    (2, 2, 3)
    >>> np.tensordot(m,np.outer(p,q))
    array([ 0.576,  0.424])
    >>> z = np.apply_along_axis(sum,1,m)
    >>> z.shape
    (2, 3)
    >>> z * q
    array([[ 0.42,  0.3 ,  0.3 ],
    [ 0.18,  0.7 ,  0.1 ]])
    >>> z.dot(q)
    array([ 1.02,  0.98])
    >>> v = z.dot(q)
    >>> v /= np.sum(v)
    >>> v
    array([ 0.51,  0.49])
    >>>  a = np.array( (.6,.4) )
    >>> b = np.array( [.6,.4] )
    >>> a == b
    array([ True,  True], dtype=bool)
    >>> np.all(a == b)
    True
    >>> np.any(a == b)
    True
    >>> np.any(np.array( [False,True] ))
    True
    >>> np.all(np.array( [False,True] ))
    False
    """
def generate_graph():
    """ un petit graphe pour vérifier que les méthodes cfc et hasCircuit sont ok """
    nodes = [str(i) for i in range(1,13)]
    pat = [[4],[1,5,7],[2],[3,5], [3,6],[10],[6,8],[9],[7,10],[],[10,12],[9,11]]
    links = [ (str(i+1),str(x))  for i in range(12) for x in pat[i] ]
    #print(nodes,len(nodes)) #13 sommets
    #print(links,len(links)) # 19 aretes http://www.christian.braesch.fr/page/recherche-de-composantes-fortement-connexes
    return graphe(nodes,links) 
    
if __name__ == '__main__' :
    DOIT = False 
    if DOIT: old_main()
