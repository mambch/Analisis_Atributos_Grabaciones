import numpy as np

def calcular_volumen(y_segmento):
    """
    Calcula el volumen (energía RMS) de un segmento de audio.
    """
    if len(y_segmento) == 0:
        return 0.0
    rms = np.sqrt(np.mean(y_segmento**2))
    return float(rms)
