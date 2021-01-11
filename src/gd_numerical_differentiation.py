import matplotlib.pyplot as plt

LEN = 128

EPOCH = 200
LEARNING_RATE = 1.0
LEARNING_RATE_DECAY_STEP = 50
LEARNING_RATE_DECAY_FACTOR = 4.0
EPSILON = 0.00001

def reponseImpulsion(A, B):
    #Définition des x[n] eq.4, plus zeros padding de 1 sur la gauche
    impulsion = [0.0] * (LEN+1)
    impulsion[1] = 1.0

    #Execution de l'eq.2 (padding pour y[n] aussi)
    y = [0.0] * (LEN+1)
    for i in range(1, LEN+1):
        y[i] = (A * impulsion[i]) + (B * y[i-1])

    #Suppression du padding, y[n] a bien la longueur 'LEN'.
    return y[1:]

#Définition des y[n] cibles (A=1.0 et B=-0.95)
CIBLE = reponseImpulsion(1.0, -0.95)

def lossMSE(y_true, y_pred):
    mse = 0
    for i in range(LEN):
        mse += (y_true[i] - y_pred[i])**2
    mse /= LEN
    return mse

def gradient(A, B):
    gradientA = 0.0
    gradientB = 0.0
    
    #Calcul l'erreur actuelle provoquée par les paramètres A et B
    lossRef = lossMSE(CIBLE, reponseImpulsion(A, B))

    #Estimation  de la dérivée de A
    A += EPSILON
    loss = lossMSE(CIBLE, reponseImpulsion(A, B))
    gradientA = (loss - lossRef) / EPSILON
    A -= EPSILON#Remet A dans son état initial
    
    #Estimation  de la dérivée de B
    B += EPSILON
    loss = lossMSE(CIBLE, reponseImpulsion(A, B))
    gradientB = (loss - lossRef) / EPSILON
    B -= EPSILON#Remet B dans son état initial
    
    return gradientA, gradientB

def main():
    global LEARNING_RATE
    
    #Définition des paramètres à optimiser
    A = 0
    B = 0
    
    #Prépare l'affichage du résultat
    true, = plt.plot(CIBLE)
    line, = plt.plot(CIBLE)
    plt.legend((true, line), ('CIBLE', 'Prédiction'))
    
    for e in range(EPOCH):
        #Optimise les paramètres A et B en fonction des gradients
        gradA, gradB = gradient(A, B)
        A -= gradA * LEARNING_RATE
        B -= gradB * LEARNING_RATE
        
        #Règle d'apprentissage facultative
        if (e+1) % LEARNING_RATE_DECAY_STEP == 0:
            LEARNING_RATE /= LEARNING_RATE_DECAY_FACTOR
        
        #Calcul pour le graphe seulement
        pred = reponseImpulsion(A, B)
        loss = lossMSE(CIBLE, pred)
        print("{0:4d} / {1:4d} : loss {2: .6f}".format(e+1, EPOCH, loss))
        #Affichage du graphe
        line.set_ydata(pred)
        plt.title('A={} B={}'.format(A, B))
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    main()
    input("Press [enter] to continue.")