import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.animation as animation


DPI = 50

def produce_movie(X, evolve, save_path, num_steps = 100, cmap = None, interpolation = 'bicubic'):

    if len(X.shape) == 2 and cmap is None:
        cmap = 'gray_r'

    fig = plt.figure(figsize=(16, 9))
    im = plt.imshow(X, cmap=cmap, interpolation=interpolation, vmin=0, vmax=1)
    plt.axis('off')
    
    def update(i):
        
        if(i%(num_steps//10)==0):
            print('Step {}/{}'.format(i, num_steps))
        
        if (i==0):
            return im,
        nonlocal X
        X = evolve(X)
        im.set_array(X)
        return im,

    ani = animation.FuncAnimation(fig, update, num_steps, interval=50, blit=True)
    ani.save(save_path, fps=25, dpi = DPI)

def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)

def growth_lenia(u):
    mu = 0.15
    sigma = 0.015
    return -1 + 2 * gauss(u,mu,sigma)

R = 13
y, x = np.ogrid[-R:R, -R:R]
distance = np.sqrt((1+x)**2 + (1+y)**2) / R

mu = 0.5
sigma = 0.15
K_lenia = gauss(distance, mu, sigma)
K_lenia[distance > 1] = 0               # Cut at d=1
K_lenia = K_lenia / np.sum(K_lenia)     # Normalize

dt = 0.1
def evolve_lenia(X):  
    U = sp.signal.convolve2d(X, K_lenia, mode='same', boundary='wrap')
    X = X + dt * growth_lenia(U)
    X = np.clip(X, 0, 1)
    return X


#gaussian dot
def gaussian_dot():
    N = 256
    M = int(np.ceil((16*N)/9))
    X = np.ones((M, N))
    radius = 36
    y, x = np.ogrid[-N//2:N//2, -M//2:M//2]
    X = np.exp(-0.5 * (x*x + y*y) / (radius*radius))
    plt.imshow(X, cmap='inferno', interpolation='none')
    produce_movie(X, evolve_lenia, '3-lenia_spot.gif', 500, cmap = 'inferno')


#orbium
def orbium():
    orbium = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
    N = 128
    M = int(np.ceil((16*N)/9))
    X = np.zeros((N, M))
    pos_x = M//6
    pos_y = N//6
    X[pos_x:(pos_x + orbium.shape[1]), pos_y:(pos_y + orbium.shape[0])] = orbium.T
    plt.imshow(X, cmap='inferno')
    produce_movie(X, evolve_lenia, '3-lenia_orbium.gif', 800, cmap = 'inferno')


def draw_initial_state(N=256, init_brush="gaussian", init_size=10):
    """
    Interface interactive pour dessiner une condition initiale.
    
    Contrôles :
      - Clique gauche (maintenir pour dessiner en continu)
      - Clique droit : efface
      - b : binaire
      - g : gaussien
      - + / - : change la taille du pinceau
      - Fermer la fenêtre pour lancer l'évolution
    """
    X = np.zeros((N, N))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(X, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Dessine avec la souris - Ferme la fenêtre pour lancer l'évolution")

    brush = {"type": init_brush, "size": init_size, "drawing": False, "erasing": False}

    def add_brush(x, y, erase=False):
        """Applique un pinceau (ou gomme) autour de (x, y)."""
        yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        d2 = (xx - x) ** 2 + (yy - y) ** 2

        if erase:
            mask = d2 < brush["size"] ** 2
            X[mask] = 0.0
        elif brush["type"] == "binary":
            mask = d2 < brush["size"] ** 2
            X[mask] = 1.0
        elif brush["type"] == "gaussian":
            X[:] += np.exp(-0.5 * d2 / (brush["size"] ** 2))
            X[:] = np.clip(X, 0, 1)

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # clic gauche → dessiner
            brush["drawing"] = True
            add_brush(int(event.xdata), int(event.ydata))
            im.set_data(X)
            fig.canvas.draw_idle()
        elif event.button == 3:  # clic droit → effacer
            brush["erasing"] = True
            add_brush(int(event.xdata), int(event.ydata), erase=True)
            im.set_data(X)
            fig.canvas.draw_idle()

    def on_release(event):
        brush["drawing"] = False
        brush["erasing"] = False

    def on_motion(event):
        if event.inaxes != ax:
            return
        if brush["drawing"]:
            add_brush(int(event.xdata), int(event.ydata))
            im.set_data(X)
            fig.canvas.draw_idle()
        elif brush["erasing"]:
            add_brush(int(event.xdata), int(event.ydata), erase=True)
            im.set_data(X)
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "b":
            brush["type"] = "binary"
            print("Brush → binary")
        elif event.key == "g":
            brush["type"] = "gaussian"
            print("Brush → gaussian")
        elif event.key == "+":
            brush["size"] += 2
            print(f"Brush size → {brush['size']}")
        elif event.key == "-":
            brush["size"] = max(1, brush["size"] - 2)
            print(f"Brush size → {brush['size']}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    return X

# L'utilisateur dessine une configuration
X = draw_initial_state(N=256)

# Lancer l’évolution
produce_movie(X, evolve_lenia, "5-lenia_drawn.gif", 300, cmap="inferno")


