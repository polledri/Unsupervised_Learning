from agent import Agent
import numpy as np
from scipy.special import softmax

def default_policy(agent: Agent,WORLD_SIZE:int=3) -> str:
    """
    Policy of the agent
    return "left", "right", or "none"
    """
    actions = ["left", "right", "none"]

    # on donne un poid à chaque case de la grille selon sa pertinance (pertinance=reward) / on peut ameliorer 
    #le code en rajoutant une meilleure pondération sur les cases qu'on a pas visité par apport à celle qu'on a visité mais sans reward
    ponderation=softmax(np.array(agent.known_rewards)) # on fait un softmax pour tt remettre entre 0 et 1 de somme 1 en gardant "les proportions"
    #calcul des poids à gauche à droite et sur moi pour savoir quel endroit est le plus pertinent à visiter
    alpha=sum(ponderation[:agent.position])
    beta=ponderation[agent.position]
    # on tire une loi uniforme sur [0,1] pour pouvoir tirer la décision tq P(action)=pondération_de_la_région
    U=np.random.rand()
    if U<alpha: # alpha=somme des pondérations à gauche
        action='left'
        print('left')
    elif U<alpha+beta: # beta= pondération de la position du joueur
        action='none'
    else:
        action='right'
        print('right')
    assert action in actions
    return action
