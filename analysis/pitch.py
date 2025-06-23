import parselmouth
import numpy as np

def analizar_pitch(y_segmento, sr):
    """
    Calcula el pitch promedio, mínimo y máximo en Hz para un segmento de audio.
    Salta el análisis si el segmento es demasiado corto.
    """
    try:
        # Verificar duración mínima
        duracion_segundos = len(y_segmento) / sr
        if duracion_segundos < 0.05:
            raise ValueError("Segmento demasiado corto para análisis de pitch.")

        # Crear objeto Sound desde array
        snd = parselmouth.Sound(y_segmento, sampling_frequency=sr)

        # Calcular pitch con límites seguros
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)

        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Eliminar silencios o frames inválidos

        if len(pitch_values) == 0:
            raise ValueError("Pitch no detectable en el segmento.")

        return {
            'pitch_mean': round(np.mean(pitch_values), 2),
            'pitch_min': round(np.min(pitch_values), 2),
            'pitch_max': round(np.max(pitch_values), 2)
        }
    except Exception as e:
        print(f"[!] Error en análisis de pitch: {e}")
        return {'pitch_mean': 0, 'pitch_min': 0, 'pitch_max': 0}
