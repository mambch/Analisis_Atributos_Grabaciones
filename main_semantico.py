import os
import json
import pandas as pd
from collections import Counter
from textstat import fernandez_huerta

from analysis.pregunta_respuesta import detectar_pregunta, detectar_respuesta
from analysis.interrupciones import detectar_interrupcion
from analysis.emociones import detectar_emocion
from analysis.palabras_clave import detectar_palabras_clave

# === CONFIGURACIÓN ===
TRANSCRIPCION_PATH = r"C:\Users\erika\S4P\salidas\transcripcion_por_segmento.json"
OUTPUT_CSV = "resultados_semantico.csv"

# === LISTA DE CONECTORES COMUNES ===
CONECTORES = {
    "además", "entonces", "o sea", "bueno", "así que", "por lo tanto", "por eso",
    "en realidad", "claro", "digamos", "también", "sin embargo", "aunque"
}

# === CARGA DE TRANSCRIPCIÓN ===
print("📥 Cargando transcripción...")
with open(TRANSCRIPCION_PATH, "r", encoding="utf-8") as f:
    segmentos = json.load(f)

resultados = []
seg_anterior = None

# === PROCESAR CADA SEGMENTO ===
print("🔍 Analizando segmentos...")
for seg in segmentos:
    texto = seg['text'].strip().lower()

    es_pregunta = detectar_pregunta(texto)
    es_respuesta = detectar_respuesta(texto)
    interrupcion = detectar_interrupcion(seg, seg_anterior)
    emocion = detectar_emocion(texto)
    palabras_clave = detectar_palabras_clave(texto)
    legibilidad = fernandez_huerta(texto) if texto else None

    # === Repeticiones
    palabras = [p for p in texto.split() if len(p) > 2]
    conteo = Counter(palabras)
    repeticiones = sum(1 for _, c in conteo.items() if c >= 2)

    # === Conectores
    conectores_detectados = [c for c in CONECTORES if c in texto]
    uso_excesivo_conectores = len(conectores_detectados) > 3

    resultados.append({
        'speaker': seg['speaker'],
        'start': round(seg['start'], 2),
        'end': round(seg['end'], 2),
        'duracion': round(seg['end'] - seg['start'], 2),
        'es_pregunta': es_pregunta,
        'es_respuesta': es_respuesta,
        'interrupcion': interrupcion,
        'emocion_detectada': emocion,
        'palabras_clave_detectadas': ", ".join(palabras_clave) if palabras_clave else "",
        'indice_legibilidad': legibilidad,
        'num_repeticiones': repeticiones,
        'conectores_usados': ", ".join(conectores_detectados),
        'uso_excesivo_conectores': uso_excesivo_conectores
    })

    seg_anterior = seg

# === GUARDAR RESULTADOS ===
df = pd.DataFrame(resultados)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Análisis semántico completado. Resultados guardados en: {OUTPUT_CSV}")



