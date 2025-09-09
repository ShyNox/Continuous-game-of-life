**Lenia Simulation **

Ce projet est une implémentation en Python d’un automate cellulaire Lenia (Life-like Continuous Cellular Automata), permettant de simuler des motifs évolutifs, générer des animations et interagir avec une interface de dessin pour définir des conditions initiales personnalisées.

**Fonctionnalités**

Évolution de Lenia à partir de noyaux et fonctions de croissance définis (Orbium, Gaussian spot…).

Génération de vidéos ou GIFs montrant l’évolution des configurations.

Interface de dessin interactive :

Dessin à la souris pour définir l’état initial

Clic gauche maintenu : dessiner

Clic droit maintenu : effacer

b → pinceau binaire (valeurs 0 ou 1)

g → pinceau gaussien (dégradé doux)

+ / - → ajuster la taille du pinceau

Export automatique de l’évolution en .gif

**Dépendances**

Le projet utilise Python 3 et les bibliothèques suivantes :

pip install numpy matplotlib scipy

**Structure du projet**

produce_movie() → génère une animation à partir d’une condition initiale et d’une fonction d’évolution.

evolve_lenia() → règle d’évolution de Lenia (convolution + fonction de croissance).

gaussian_dot() → exemple avec un point gaussien initial.

orbium() → exemple avec un pattern Orbium.

draw_initial_state() → interface interactive pour dessiner un état initial.

**Exemple d’utilisation**
1. Lancer un Orbium prédéfini
orbium()

2. Dessiner une configuration à la main
X = draw_initial_state(N=256)
produce_movie(X, evolve_lenia, "lenia_drawn.gif", 300, cmap="inferno")

**Résultats**

Exemple Orbium :


Exemple dessin manuel (export .gif) :
Image personnalisée générée par l’utilisateur.

**Options à modifier**

N : taille de la grille (ex. N=256)

brush : type de pinceau initial (binary ou gaussian)

num_steps : nombre d’itérations de l’évolution

cmap : colormap matplotlib (inferno, gray_r, etc.)

**Références**

Lenia: Biology of Artificial Life
 — Bert Wang-Chak Chan

Wikipedia - Lenia

**Licence**

Projet libre à usage personnel et éducatif.
