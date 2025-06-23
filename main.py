import os
import json
import librosa
import pandas as pd
from analysis.volumen import calcular_volumen
from analysis.velocidad import calcular_velocidad
from analysis.pitch import analizar_pitch
from analysis.emocion_audio import detectar_emocion_audio  # ✅ NUEVO

# === CONFIGURACIÓN DE RUTAS ===
AUDIO_PATH = r"C:\Users\erika\OneDrive\Escritorio\S4P\Ventas_wav\AP+AHORRO_20250502_ADA.GARCIA_19430567.wav"
SEGMENTOS_PATH = r"C:\Users\erika\S4P\salidas\transcripcion_por_segmento.json"
DIARIZACION_PATH = r"C:\Users\erika\S4P\salidas\diarizacion.json"
OUTPUT_CSV = "resultados_analisis.csv"

# === CARGA DE AUDIO ===
print("📥 Cargando audio...")
y, sr = librosa.load(AUDIO_PATH, sr=None)

# === CARGA DE TRANSCRIPCIÓN Y DIARIZACIÓN ===
print("📥 Cargando segmentos...")
with open(SEGMENTOS_PATH, "r", encoding="utf-8") as f:
    segmentos = json.load(f)

print("📥 Cargando diarización...")
with open(DIARIZACION_PATH, "r", encoding="utf-8") as f:
    diarizacion = json.load(f)

# === PROCESAR SEGMENTOS ===
resultados = []

for i, seg in enumerate(segmentos):
    start = seg['start']
    end = seg['end']
    speaker = seg['speaker']
    texto = seg['text'].strip()

    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_segmento = y[start_sample:end_sample]

    volumen = calcular_volumen(y_segmento)
    velocidad = calcular_velocidad(texto, start, end)
    pitch_data = analizar_pitch(y_segmento, sr)
    emocion_audio = detectar_emocion_audio(AUDIO_PATH, start, end)  # ✅ NUEVO

    # === SILENCIO PREVIO ===
    if i == 0:
        silencio_previo = 0.0
    else:
        silencio_previo = round(start - segmentos[i - 1]['end'], 2)
        silencio_previo = max(0.0, silencio_previo)  # sin negativos

    es_silencio_largo = silencio_previo > 1.5

    resultados.append({
        'speaker': speaker,
        'start': round(start, 2),
        'end': round(end, 2),
        'duracion': round(end - start, 2),
        'volumen_rms': round(volumen, 6),
        'velocidad_palabras_seg': velocidad,
        'pitch_mean_hz': pitch_data['pitch_mean'],
        'pitch_min_hz': pitch_data['pitch_min'],
        'pitch_max_hz': pitch_data['pitch_max'],
        'num_palabras': len(texto.split()),
        'texto': texto,
        'tiempo_silencio_previo': silencio_previo,
        'es_silencio_largo': es_silencio_largo,
        'emocion_por_audio': emocion_audio  # ✅ NUEVA COLUMNA
    })

# === CONVERTIR A DATAFRAME ===
df = pd.DataFrame(resultados)

# === CALCULAR % DE HABLA POR SPEAKER ===
total_por_speaker = df.groupby('speaker')['duracion'].sum()
total_global = total_por_speaker.sum()
porcentaje_habla = (total_por_speaker / total_global * 100).round(2)

# Asignar a cada fila
df['porcentaje_habla'] = df['speaker'].map(porcentaje_habla)

# === GUARDAR RESULTADOS ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Análisis completado. Resultados guardados en: {OUTPUT_CSV}")


