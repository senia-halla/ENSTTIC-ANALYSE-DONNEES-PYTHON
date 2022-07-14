############# TP2 : Structuration de données et calcul matriciellles 
import numpy as np
from numpy.lib.index_tricks import fill_diagonal #Importer la bibliotheque numpy 
import scipy as sc #importer la bibliotheque scipy 
import matplotlib as mpl #importer la bibliotheque matplotlib 
import math
####################################################
#Creation de tableau multidimensionnel 

#Creation des vecteurs et des matrices : d'un seule type 
v1 = np.array([1.5,5.3,25.5,9])
print(v1)
#[ 1.5  5.3 25.5  9. ] : vecteur de float

v2 = np.array([1, 3, 2])
print(v2)
#[1 3 2] : vecteur d'entier 

v3 = np.array([1, 3, 2], dtype=float)
print(v3)
#[1. 3. 2.] : vecteur d'entier convertie en float 
m = np.array ( [[3.2,2.5],[1.2,4.8],[1.5,6.3]] )
print(m)
#[[3.2 2.5]
# [1.2 4.8]
# [1.5 6.3]]

#type des vecteur/matrice
print(type(v1)) #<class 'numpy.ndarray'> => Multidimensionnal array 
print(type(m))  #<class 'numpy.ndarray'> => Multidimensionnal array 

#type de base : type du contenu 
print(v2.dtype) #int32
print(m.dtype) #float64

#dimension de tab 
print(v1.ndim) #1 => 1 dimension : colonne 
print(m.ndim)  #2 => 2 dimmensions : lignes et colonnes 

#nombre d'element de chaque dimension : la taille du tableau:
print(v1.shape) #(4,) : 4 colonnes (vecteur une seule ligne )
print(v2.shape) #(3,) : 3 colonnes 
print(m.shape) #(3,2) : 3 ligne 2 colone 

v4 = np.array([[1],[2],[3]])
print(v4.ndim) #2 => 2 dimension (une matrice de plusieurs ligne et une seule colonne )
print(v4.shape) #(3,1) : 3 ligne 1 colonne


#taille de tab : nombre d'elements totale 
print(v1.size) # 4 elements : 4x1
print(m.size)  # 6 elements : 3x2

#Creation d'un tableau d'objet
v = np.array([{"Alger":(16,3000000)},{"Oran":(31,2000000)}]) 
print(v) 
#[{'Alger': (16, 3000000)} {'Oran': (31, 2000000)}]
print(v.dtype) #object
print(v.ndim)  #1
print(v.shape) #(2,)
print(v.size)  #2

###########################################################
#Creation des tab mul avec la fonction arrange : la valeur du stop n'est pas incluse
#step : indique le pas 
u1 = np.arange(start=0,stop=10)
print(u1)
#[0 1 2 3 4 5 6 7 8 9]

u2 = np.arange(start=0, stop=10, step=2)
print(u2)
#[0 2 4 6 8]

m=np.arange(0,10).reshape(2,5) # creation d'un matrice de 2 ligne et 5 colonnes 
print(m)
#[[0 1 2 3 4]
# [5 6 7 8 9]]

#La methode reshape : change la structure d'un tab en une autre structure 
#vec en mat ou bien mat en vec

############################################################
#Creation avec la fonction linspace : ici la valuer du stop incluse 
#num : indique le nombre d'element 
l = np.linspace(start=0,stop=10,num=5)
print(l)
#[ 0.   2.5  5.   7.5 10. ]

######################################################################
#Creation de tab a valeurs identiques 
o1 = np.ones(shape=3)
print(o1)
#[1. 1. 1.]

o2=np.zeros(shape=3)
print(o2)
#[0. 0. 0.]

o3 = np.full(shape=3,fill_value=1.2)
print(o3)
#[1.2 1.2 1.2]

o4 = np.full(shape=(2,4),fill_value=3)
print(o4)
#[[3 3 3 3]
# [3 3 3 3]]

##############################################################
#Ajout, supression, insertion et redimensionnement 
#Fonctions Append et Delete 
s1 = np.array([1.5, 5.3, 25.5, 9])
s11 = np.append(s1,10) #Ajout de l'elemenet 10 a la fin du vecteur s1
print(s1) #[ 1.5  5.3 25.5  9. ]
print(s11) #[ 1.5  5.3 25.5  9.  10. ]

s2 = np.array([11,12])
s22 = np.append(s1,s2) #Concatenation de deux vecteurs 
print(s22) #[ 1.5  5.3 25.5  9.  11.  12. ]

s22_delete = np.delete(s22,3) #supression de l'element d'indice 3
print(s22_delete) #[ 1.5  5.3 25.5 11.  12. ]

#Redimensionnemt d'un vecteur 
s3 = np.array([1,2,3])
print(s3) #[1 2 3]
s3.resize((5)) # les nouvelles cases seront remplie de 0
print(s3) #[1 2 3 0 0]

#Ajout d'une ligne ou d'une colonne a une matrice 
m = np.array([[1.2,2.5],[3.2,1.8],[1.1,4.3]])
v1 = np.array([[4.1,2.6]])
v2 = np.array([[4],[5],[6]])
m1 = np.append(m,v1,axis=0) #Ajout d'une ligne 
m2 = np.append(m,v2,axis=1) #Ajout d'une colonne 
print(m)
#[[1.2 2.5]
# [3.2 1.8]
# [1.1 4.3]]

print(m1)
#[[1.2 2.5]
# [3.2 1.8]
# [1.1 4.3]
# [4.1 2.6]]

print(m2)
#[[1.2 2.5 4. ]
# [3.2 1.8 5. ]
# [1.1 4.3 6. ]]

#Inserer une ligne é un endroit définit 
m3 = np.insert(m,2,v1,axis=0) #Ajout du v1 a la ligne 3 (0 1 2)
print(m3)
#[[1.2 2.5]
# [3.2 1.8]
# [4.1 2.6]
# [1.1 4.3]]

# Supression d'une colone à un endroit précis 
m4 = np.delete(m,1,axis=1) #Supression de la 2 eme colonne 
print(m4)
#[[1.2]
# [3.2]
# [1.1]]

#Redimensionnement de la matrice 
h = np.resize(m,new_shape=(2,3))
print(h)
#[[1.2 2.5 3.2]
# [1.8 1.1 4.3]]


###########################################################################
#Extraction de plages de valeurs : Les sous Tableaux 
v = np.array([7,2.3,5.2,3.6,1.9])
print(v[:]) # Affichage de tout les éléments de v
print(v[0]) # Affichage du premier élément du vecteur 7.0
print(v[-1]) # Affichage du dernier élément du vecteur 1.9
print(v[1:3]) # Affichage d'une plage d'elements :  2.3 -> 5.2 ([m:n] de l'element m jusqu'a n-1)
print(v[:3]) #Affichage des 3 1er element 7,2.3,5.2
print(v[2:]) # Afichage des (5-2) dernier elements 5.2,3.6,1.9
print(v[-3:]) #affichage des 3 derniers elements 5.2,3.6,1.9
print(v[1:4:2]) #Afichage de 3 element a partir de 1 AVEC un saut de 2 2.3 3.6
print(v[3:0:-1]) #  Affichage en arriere (pas de -1 )3.6 1.9 7
print(v[::-1]) # 1.9 3.6 5.2 2.3 7.0 


##########################################################################
#Pour les matrice 
m = np.array([[1,2],[3,4],[5,6]])
print(m[:,:]) 
# Affichage de tout les elements 
#[[1 2]
# [3 4]
# [5 6]]
print(m[0,0])   #affiche 0,0)=1
print(m[-1,-1]) #affiche len(m),len(m))=6
print(m[0:2,:]) #affiche des 2-0 premiere lignes
#1 2
#3 4
print(m[:2,:]) #affiche les 2-0 premiere ligne : vide => debut
#1 2
#3 4
print(m[1:,:]) #affiche len - 1 ligne a partir de celle d'indice 1
#3 4
#5 6
print(m[-1,:]) #affichage de tout les elemnts de la derniere lignes 
#5 6
print(m[-2:,:]) # affichage des 2 dernieres lignes 
#3 4
#5 6 

####################################################################
#Extraction en utilisant les conditions : 
u = np.array([1,2,3,4,5,6,7,8,9,10])
b = u < 5
print(b) #[ True  True  True  True False False False False False False]¨
print(u[b]) #[1 2 3 4]  || print(u[u<5])

n = np.array([[1,2],[3,4],[5,6]])
b = np.array([True,False,True],dtype=bool)
print(m[b,:]) # affichage de toutes les lignes correspendantes a true 
# 1 2
# 5 6

##################################################################
#Exercice :
w = np.array([[1,2],[3,4],[5,6]])

# 1. Calculer la somme des lignes : 
somme_ligne = np.sum(m,axis=1) #axis=1 => ligne
print(somme_ligne) #3 7 11

# 2.Calculer le min de somme_ligne 
min_somme = np.min(somme_ligne)
print(min_somme) #3

# 3. Vecteur de bool permetant de recupere les lignes correspandante au min 
bool = (somme_ligne == min_somme)
print(bool) #[ True False False]

# 4. Afficher la ligne dans la somme = min (Filtre Bool)
min_ligne = w[bool,:]
print(min_ligne) #[1,2]

###################################################################
#Tri et Recherches 
v = np.array([7,2.3,5.2,3.6,1.9]) 
m = np.array([[1.2,2.5],[3.2,1.8],[1.1,4.3]])
print(np.max(v)) # 7.
print(np.argmax(v)) #0 : 1 er element 
print(np.max(m,axis=0)) #les maximum de chaque colonne [3.2 4.3]
print(np.argmax(m,axis=0)) #[1 2] : 2eme element de 1 er colonne et 3 eme element de 2 eme colonne 
print(np.min(m,axis=1)) #les minimum de chaque ligne  [1.2 1.8 1.1]
print(np.argmin(m,axis=1)) #[0 1 0] : 1er element de 1er ligne , 2 eme element de 2 eme ligne , 1 er element de 3 eme ligne 

#Recherche d'une valeur précise : 
print(np.where(v==3.6)) #(array([3], dtype=int64),)
print(np.where((m>2)&(m<4))) #(array([0, 1], dtype=int64), array([1, 0], dtype=int64))

#Tri :
print(np.sort(v)) #Tri accendant [1.9 2.3 3.6 5.2 7. ]
print(np.argsort(v)) #Indices des elements triés [4 1 3 2 0]
print(np.sort(m,axis=0)) #Tri par rapport au colonne 
#[[1.1 1.8] #min des deux colonnes
#[1.2 2.5]
#[3.2 4.3]] #max des deux colonnes 

a = np.array([1,2,2,1,1,2]) #les valeurs uniques sans repetition 
print(np.unique(a)) #[1 2]

#############################################################################
#Opérations ensemblistes :
v1 = np.array([1,2,5,6])
v2 = np.array([2,1,7,4])

print(np.intersect1d(v1,v2)) #[1 2] : Intersection v1 ET v2
print(np.union1d(v1,v2)) #[1 2 4 5 6 7] : Union v1 OU v2
print(np.setdiff1d(v2,v1)) #[4 7] : element se trouvant dans v2 ET NON PAS v1
print(np.setdiff1d(v1,v2)) #[5 6] : element se trouvant dans v1 ET NON PAS v2

##############################################################################
#Calcul Matricielle : 

#1. Calcul entre vecteurs élément par élément : meme size 
print(v1+v2)   #[ 3  3 12 10] 
print(v1*v2)   #[ 2  2 35 24]
print(2*v1)    #[ 2  4 10 12]
print(v1 > v2) #[False  True False  True]

#2. Norme et Produit scalaire : 
v1 = np.array([1.2,1.3,1.0])
v2 = np.array([2.1,0.8,1.3])
p = np.vdot(v1,v2) #produit scalaire || np.sum(v1*v2) 
print(p) #4.86
n = np.linalg.norm(v1) #La norme d'un vecteur || math.sqrt(np.sum(v1**2))
print(n) #2.03

# Transposé, Produit matriciel, Déterminant et Matrice inverse 
m1 = np.array([[1.2,2.5],[3.2,1.8],[1.1,4.3]])
m2 = np.array([[2.1,0.8],[1.3,2.5]])
transpose_m1 = np.transpose(m1) #transposé
print(transpose_m1) # m1 = 3x2 m1_transpose = 2x3
#[[1.2 3.2 1.1]
# [2.5 1.8 4.3]]
prod_mat_m1_m2 = np.dot(m1,m2) #np.dot(m1,m2) =/= np.dot(m2,m1)
print(prod_mat_m1_m2) # m1=3x2 2x2=m2 => m1_m2 = 3x2
#[[ 5.77  7.21]
# [ 9.06  7.06]
# [ 7.9  11.63]]
detrminant_m2 = np.linalg.det(m2)
print(detrminant_m2) 
#4.210000000000001
inverse_m2 = np.linalg.inv(m2)
print(inverse_m2)
#[[ 0.59382423 -0.19002375]
# [-0.3087886   0.49881235]]

######################################################################
#Résolution d'equation : M.x = y : On cherche les valueur du vecteurs x 
M = np.array([[2.1,0.8],[1.3,2.5]])
y = np.array([1.7,1.0])
print(np.linalg.solve(M,y)) #[ 0.81947743 -0.02612827]
print(np.dot(np.linalg.inv(M),y)) #Verification [ 0.81947743 -0.02612827]

#Matrice symétriue avec transposé : 
S = np.dot(np.transpose(M),M)
print(S)
#[[6.1  4.93]
# [4.93 6.89]]

#Valeurs et vecteurs propres d'une matrice symétrique
print(np.linalg.eigh(S))
#(array([ 1.54920128, 11.44079872]), array([[-0.73480125,  0.67828248],
       #[ 0.67828248,  0.73480125]]))