######## TP1 - Python : Prise en main 
import math #importer un module 

############################################################
#Variable
x = 2
y = 3.14
print(x,y)

Hello_1 = 'Hello'
Hello_2 = "Hello"
Hello_3 = '''Hello'''
Hello_4 = """Hello"""
print(Hello_1,Hello_2,Hello_3,Hello_4)

###################################################################
#Operations sur les chaine de caractere 
chaine_1 = " Hello "
chaine_2 = " World."
Hello_World = chaine_1 + chaine_2
print(Hello_World)

Hello_World_Rep = x*Hello_World
#Hello_World_Rep = y*Hello_World : ne marchera pas => une chaine de caractere peut etre multiplier qu'aver un nombre entier 
print(Hello_World_Rep)

####################################################################
#La fonction Type : 
print(type(x)) # <class 'int'>
print(type(y)) # <class 'float'>
print(type(Hello_1)) # <class 'str'>
print(type(chaine_1)) # <class 'str'>

##############################################################
#Les fonction de conversion 
print(str(x)) # '2'
print(float(x)) # 2.0
z = '2573771' 
print(int(z)) # 2573771
#n = 'Senia Halla'
#print(int(n))
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#ValueError: invalid literal for int() with base 10: 'Halla Senia'

################################################################
# La fonction Print : 
print('Hello', end="") # paaremetre end :  pas de retour a la ligne 
print('World')

#sep et end : 
prix = 6000
article = 'PC '
q = '?'
print(article,prix, " DA ", sep="") # parametre sep : pas de separation entre les variable 
#PC 6000 DA
print(article,prix, " DA ",q, sep="-") # separation - entre les varible  
#PC -6000- DA -?

# => la methode d'objet format 
# {} ce sont des reservation 
print("{} : {} DA".format(article,prix)) #PC  : 6000 DA
print("{0} : {1} DA".format(article,prix)) #PC  : 6000 DA
print("{1} : {0} DA".format(article,prix)) #6000  : PC DA

############################################################
# Utilisation du module math : 
pi = math.pi 
print("Pi = ",pi) #Pi =  3.141592653589793
print("Pi avec 2 chiffres apres virgule {:.2f}".format(pi)) # 3.14
print("Pi avec 3 chiffres apres virgule {:.3f}".format(pi)) # 3.142 : il fait l'arrondissement 
print("pi = {:.2f} , exp = {:.2f}".format(pi,math.e)) #pi = 3.14 , exp = 2.72

#############################################################
# Affichage avec nombre de positions :
print(10)
#d : decimal
print("{:>6d}".format(10))
print("{:<6d}".format(10))
print("{:^6d}".format(10))
print("{:¬^6d}".format(10))
print("{:%<6d}".format(10))
print("{:&>6d}".format(10))

#xxxxxx : 6 positions 
#10
#    10 : ajustement de 6 position a droite
#10     : ajustement de 6 poristion a gauche     
#  10   : ajustement au centre de 6 positions 
#¬¬10¬¬
#10%%%%
#&&&&10

#s : str
print("Article 1  {:>7s}".format("PC"))
print("Article 1  {:>7s}".format("CLAVIER"))
#aaaaaaa 1234567
#Article      PC 
#Article CLAVIER

#f : float
print("{:7.3f}".format(math.pi))
print("{:10.3f}".format(math.pi))
#1234567890
#  3.142
#     3.142

###############################################################################
#Saisie avec la fonction input : c'est une fonction bloquante (Le programme restera bloqué tant que l'utilisateur n'a pas saisie un contenu)
#a = float(input('Entrez Un Nombre : '))
a=5
print('Le Carré De {} Vaut {}'.format(a,a**2))


#####################################################################
#Caractere speciaux :
 
print("Une chaîne\n\tsur plusieurs lignes\n\t\tavec tabulations")
#Une chaîne
#       sur plusieurs lignes
#                avec tabulations

print("Une chaîne avec des \"guillemets\".")
#Une chaîne avec des "guillemets".


#####################################################################
#Les chaines de caractere 
nom = "Halla SeniA"
print("La taille de mon nom {} en comptant les espaces : {}".format(nom,len(nom)))
print(nom[0]) # H : affichage du premier caractere 
print(nom[-1]) # A : le dernier carctere 
print(nom[len(nom)-1],nom[-len(nom)]) #A H
print(nom[5]) # espace 

#####################################################################
#Structure conditionnelle : if elif else 
#Python structure en alignement , java en accolade , matlab end 
f = 10
g = 5 

if g < f:
      print("g est inferieur a f")
elif g > f:
      print("g est superieur a f")
else:
      print("g et f sont egaux ")

#####################################################################
#Les boucles: 
#1 - La boucle for 
s = "Parcours de chaîne"
for v in s : 
      print(v,end=" ")
#P a r c o u r s d ' u n e c h a i n e
print('\n')
#La fonction range() : range(3) = 0 1 2 
for k in range(3) : 
      print(k,end="-")
print('\n')
#0 1 2 


#range(5,10) :   # afficher chiffre de 5 a 9  (incrementation par defaut 1 )
for k1 in range(5, 10) :
      print(k1,end=" ")
#5 6 7 8 9
print('\n')

# range(0, 10,2) afficher chiffre de 0 a 9 avec un incrementation de 2 
for k2 in range(0, 10,2) :  
      print(k2,end=" ")
#0 2 4 6 8
print('\n')


#2- La boucle while : 
k3 = 0
while k3 < 5:
      print(2*k3,end=" ")
      k3+= 1
#0 2 4 6 8
print('\n')

#################################################################
#Les fonctions :
#création de la fonction 
def compteur(deb=0,fin=10,pas=1):
      i=deb
      while i<fin:
            print(i,end=' ')
            i += pas

#Appel de la foncion 
compteur()
print('\n')
#0 1 2 3 4 5 6 7 8 9

compteur(5,15,3)
print('\n')
#5 8 11 14

compteur(pas=3)
print('\n')
#0 3 6 9

compteur(10,20)
print('\n')
#10 11 12 13 14 15 16 17 18 19
