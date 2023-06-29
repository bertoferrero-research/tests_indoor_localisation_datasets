import numpy as np

def particle_filter(measurements, num_particles):
    num_timesteps = len(measurements)
    positions_particles = []
    positions = []

    # Inicialización de partículas aleatorias
    particles = np.random.rand(num_particles, 2) * 10.0  # Rango de coordenadas (0-10)

    for t in range(num_timesteps):
        # Actualización de partículas
        particles += np.random.randn(num_particles, 2) * 0.1  # Movimiento aleatorio

        # Cálculo de pesos
        errors = np.linalg.norm(particles - measurements[t], axis=1)
        weights = np.exp(-errors)

        # Muestreo de partículas
        weights /= np.sum(weights)
        indices = np.random.choice(num_particles, size=num_particles, replace=True, p=weights)
        particles = particles[indices]

        # Almacenamiento de partículas seleccionadas
        positions_particles.append(particles.copy())
        positions.append(particles.mean(axis=0))

    return positions_particles, positions