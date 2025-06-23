import parselmouth
import numpy as np

def detectar_emocion_audio(audio_path, start, end):
    try:
        snd = parselmouth.Sound(audio_path).extract_part(from_time=start, to_time=end, preserve_times=True)

        pitch = snd.to_pitch()
        mean_pitch = pitch.selected_array['frequency'].mean()
        intensity = snd.to_intensity().values[1].mean()

        # Heurística simple (puede mejorarse luego)
        if intensity > 65 and mean_pitch > 200:
            return "positiva"
        elif intensity < 50 and mean_pitch < 150:
            return "negativa"
        else:
            return "neutra"

    except Exception as e:
        print(f"[!] Error en análisis acústico de emoción: {e}")
        return "neutra"
