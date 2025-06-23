def calcular_velocidad(texto, start, end):
    """
    Calcula la velocidad de habla como palabras por segundo.
    """
    duracion = end - start
    if duracion <= 0:
        return 0.0

    num_palabras = len(texto.strip().split())
    velocidad = num_palabras / duracion
    return round(velocidad, 2)
