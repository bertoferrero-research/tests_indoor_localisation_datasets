import random
import numpy as np

def generar_trayectoria(dim_x, dim_y, margen, duracion, velocidad_media, velocidad_maxima, velocidad_minima, velocidad_muestreo, origen_x: None, origen_y: None, direccion_x: None, direccion_y: None):
    # Punto de origen aleatorio dentro del margen del terreno
    if origen_x is None:
        origen_x = random.uniform(margen, dim_x - margen)
    if origen_y is None:
        origen_y = random.uniform(margen, dim_y - margen)
    
    # Vector inicial aleatorio
    if direccion_x is None:
        direccion_x = random.uniform(-1, 1)
    if direccion_y is None:
        direccion_y = random.uniform(-1, 1)
    
    # Normalizar el vector de dirección
    magnitud = (direccion_x**2 + direccion_y**2) ** 0.5
    direccion_x /= magnitud
    direccion_y /= magnitud
    
    # Inicialización de las listas de puntos x, y y tiempo
    time = [0]
    x = [origen_x]
    y = [origen_y]
    
    tiempo_actual = 0
    
    while tiempo_actual < duracion:
        # Velocidad aleatoria entre máximo y mínimo
        velocidad = random.uniform(velocidad_minima, velocidad_maxima)
        
        # Ajuste de la velocidad para acercarse a la velocidad media
        velocidad_ajustada = (velocidad + velocidad_media) / 2
        
        # Incremento de posición basado en la velocidad ajustada
        delta_x = velocidad_ajustada * direccion_x * velocidad_muestreo
        delta_y = velocidad_ajustada * direccion_y * velocidad_muestreo
        
        # Verificar si se alcanza el margen
        while (x[-1] + delta_x > dim_x - margen) or (x[-1] + delta_x < margen) or (y[-1] + delta_y > dim_y - margen) or (y[-1] + delta_y < margen):
            # Cambiar vector de dirección
            direccion_x = random.uniform(-1, 1)
            direccion_y = random.uniform(-1, 1)
            
            # Normalizar el nuevo vector de dirección
            magnitud = (direccion_x**2 + direccion_y**2) ** 0.5
            direccion_x = direccion_x / magnitud
            direccion_y = direccion_y / magnitud
        
            # Calculamos el incremento final con la nueva dirección
            delta_x = velocidad_ajustada * direccion_x * velocidad_muestreo
            delta_y = velocidad_ajustada * direccion_y * velocidad_muestreo
        
        # Actualizar posición
        x.append(x[-1] + delta_x)
        y.append(y[-1] + delta_y)
        
        # Actualizar tiempo
        tiempo_actual += velocidad_muestreo
        time.append(tiempo_actual)
    
    return time, x, y

def add_noise_to_track(x, y, max_x_deviation: float, min_x_deviation: float, average_x_deviation: float, min_y_deviation: float, max_y_deviation: float, average_y_deviation: float):
    #Sacamos la longitud de las listas
    len_x = len(x)
    len_y = len(y)
    
    #Creamos las listas de puntos x e y con ruido aleatorio
    x_noise = [random.uniform(min_x_deviation, max_x_deviation) for i in range(len_x)]
    y_noise = [random.uniform(min_y_deviation, max_y_deviation) for i in range(len_y)]

    #Sumamos las dos listas
    x_sum = np.sum(x_noise)
    y_sum = np.sum(y_noise)

    #Calculamos la diferencia global con la media
    x_diff = average_x_deviation * len_x - x_sum
    y_diff = average_y_deviation * len_y - y_sum

    #Obtenemos el factor a sumar al array para ajustar la media
    x_average_factor = x_diff / len_x
    y_average_factor = y_diff / len_y

    #Creamos el nuevo array. Sumamos la posición original al factor y multiplicamos por un cambio de dirección aleatoria
    x_final = [ x[i] + ((x_noise[i] + x_average_factor) * random.choice([-1, 1])) for i in range(len_x)]
    y_final = [ y[i] + ((y_noise[i] + y_average_factor) * random.choice([-1, 1])) for i in range(len_y)]

    #Devolvemos
    return x_final, y_final