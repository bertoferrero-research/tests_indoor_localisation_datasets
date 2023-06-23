import numpy as np

'''
En este ejemplo, se utiliza un modelo de movimiento simple que agrega un ruido aleatorio a las partículas en cada iteración. La función predict_next_state() se encarga de actualizar las partículas utilizando este modelo de movimiento.

La función calculate_weights() se utiliza para calcular el peso de cada partícula en función de la distancia a la medida actual.

'''

def monte_carlo_filter(measurements, num_samples):
    num_measurements = len(measurements)
    
    # Inicializar partículas aleatoriamente
    particles = np.random.uniform(0, 20, size=(num_samples, 2))
    
    positions_particles = []
    positions = []
    
    for t in range(1, num_measurements):
        # Predecir el siguiente estado utilizando un modelo de movimiento simple
        particles = predict_next_state(particles)
        
        # Calcular el peso de cada partícula utilizando las medidas
        weights = calculate_weights(particles, measurements[t])
        
        # Normalizar los pesos
        weights /= np.sum(weights)
        
        # Resampling (re-muestreo) de las partículas
        indices = np.random.choice(np.arange(num_samples), size=num_samples, p=weights)
        particles = particles[indices]
        
        # Agregar las partículas seleccionadas a la lista de posiciones
        positions_particles.append(particles.copy())
        positions.append(particles.mean(axis=0))
    
    return positions_particles, positions

def predict_next_state(particles):
    # Modelo de movimiento
    num_particles = len(particles)

    #simple: agregar un ruido aleatorio a las partículas
    noise = np.random.normal(0, .25, size=(num_particles, 2))
    particles += noise

    #Test 1, ruido solo en el eje x
    #noise = np.random.normal(0, 0.1, size=(num_particles, 1))
    #particles[:,0] += noise[:,0]

    #Test 2, ruido solo en el eje y
    #noise = np.random.normal(0, 0.1, size=(num_particles, 1))
    #particles[:,1] += noise[:,0]

    
    return particles

def calculate_weights(particles, measurement):
    # Calcular el peso de cada partícula utilizando la distancia a la medida
    distances = np.linalg.norm(particles - measurement, axis=1)
    weights = 1.0 / (distances + 1e-8)  # Evitar división por cero
    
    return weights