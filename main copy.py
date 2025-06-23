import os
import json
import librosa
import pandas as pd

from analysis.volumen import calcular_volumen
from analysis.velocidad import calcular_velocidad
from analysis.pitch import analizar_pitch

# === CONFIGURACIÓN DE RUTAS ===
AUDIO_PATH = r"C:\Users\erika\OneDrive\Escritorio\S4P\Ventas_wav\AP+AHORRO_20250502_ADA.GARCIA_19430567.wav"
SEGMENTOS_PATH = r"C:\Users\erika\S4P\salidas\transcripcion_por_segmento.json"
OUTPUT_CSV = "resultados_analisis.csv"

# === VALIDACIÓN DE ARCHIVOS ===
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de audio en: {AUDIO_PATH}")
if not os.path.exists(SEGMENTOS_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo de transcripción en: {SEGMENTOS_PATH}")

# === CARGA DE AUDIO ===
print("📥 Cargando audio...")
y, sr = librosa.load(AUDIO_PATH, sr=None)

# === CARGA DE SEGMENTOS DE TRANSCRIPCIÓN ===
print("📥 Cargando segmentos...")
with open(SEGMENTOS_PATH, "r", encoding="utf-8") as f:
    segmentos = json.load(f)

resultados = []

# === PROCESAR CADA INTERVENCIÓN ===
print("🔍 Procesando segmentos...")
for seg in segmentos:
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
        'texto': texto
    })

# === GUARDAR RESULTADOS ===
df = pd.DataFrame(resultados)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Análisis completado. Resultados guardados en: {OUTPUT_CSV}")
