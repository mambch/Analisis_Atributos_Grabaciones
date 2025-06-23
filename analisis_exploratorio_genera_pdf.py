import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os

# === CONFIGURACIÓN ===
sns.set(style="whitegrid")
plt.rcParams["axes.titlesize"] = 12
os.makedirs("figs", exist_ok=True)

# === CARGAR Y UNIR DATASETS ===
df_analisis = pd.read_csv("resultados_analisis.csv")
df_semantico = pd.read_csv("resultados_semantico.csv")
df_semantico = df_semantico.drop(columns=["speaker", "start", "end", "duracion", "texto"], errors="ignore")
df = pd.concat([df_analisis, df_semantico], axis=1)

# === GENERAR GRÁFICOS Y GUARDAR FIGURAS ===
def save_plot(fig_id, title, plot_func):
    plt.figure()
    plot_func()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figs/{fig_id}.png")
    plt.close()

save_plot("velocidad", "Velocidad de habla por speaker", lambda: sns.boxplot(x="speaker", y="velocidad_palabras_seg", data=df))
save_plot("volumen", "Volumen RMS por speaker", lambda: sns.boxplot(x="speaker", y="volumen_rms", data=df))
save_plot("pitch", "Pitch medio por speaker", lambda: sns.boxplot(x="speaker", y="pitch_mean_hz", data=df))
save_plot("emociones", "Emociones detectadas", lambda: sns.countplot(x="emocion_detectada", data=df))
save_plot("interrupciones", "Interrupciones por speaker", lambda: sns.countplot(x="speaker", data=df[df["interrupcion"] == True]))

# Palabras clave
palabras_series = df["palabras_clave_detectadas"].dropna().astype(str)
palabras = palabras_series.apply(lambda x: [p.strip() for p in x.split(",") if p.strip()])
flat_list = [palabra for sublist in palabras for palabra in sublist]
top_palabras = Counter(flat_list).most_common(10)
if top_palabras:
    palabras, counts = zip(*top_palabras)
    plt.figure()
    sns.barplot(x=list(counts), y=list(palabras))
    plt.title("Top 10 palabras clave detectadas")
    plt.tight_layout()
    plt.savefig("figs/palabras_clave.png")
    plt.close()

# === GENERAR PDF ===
c = canvas.Canvas("informe_analisis_voz.pdf", pagesize=letter)
width, height = letter
y = height - inch

def add_title(text):
    global y
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, text)
    y -= 20

def add_text(text):
    global y
    c.setFont("Helvetica", 10)
    text_object = c.beginText(50, y)
    for line in text.strip().split("\n"):
        text_object.textLine(line)
        y -= 14
        if y < 100:
            c.drawText(text_object)
            c.showPage()
            y = height - inch
            text_object = c.beginText(50, y)
    c.drawText(text_object)

def add_image(path):
    global y
    if y < 300:
        c.showPage()
        y = height - inch
    c.drawImage(path, 50, y - 220, width=400, height=300)
    y -= 240

# === PÁGINA 1: Introducción ===
add_title("Informe de Análisis Exploratorio de Grabaciones")

texto_intro = """
Este informe resume los atributos acústicos y semánticos extraídos de una llamada de ventas.

Atributos evaluados:
• duracion: duración del turno de habla (en segundos).
• velocidad_palabras_seg: fluidez del hablante (palabras por segundo).
• volumen_rms: nivel de energía promedio del audio.
• pitch_mean_hz: tono medio de la voz (en Hz).
• es_pregunta / es_respuesta: si el turno es una pregunta o afirmación breve.
• emocion_detectada: sentimiento inferido del texto (positiva, negativa, neutra).
• interrupcion: si el turno comienza antes de que termine el anterior.
• palabras_clave_detectadas: conceptos comerciales clave mencionados en el turno.
"""
add_text(texto_intro)

# === PÁGINA 2: Tablas resumen ===
c.showPage()
y = height - inch

add_title("Promedios por speaker (duración, velocidad, volumen):")
agrupado = df.groupby("speaker")[["duracion", "velocidad_palabras_seg", "volumen_rms"]].mean().round(2)
for speaker, fila in agrupado.iterrows():
    add_text(f"{speaker}: duración={fila['duracion']}s, velocidad={fila['velocidad_palabras_seg']} p/s, volumen={fila['volumen_rms']}")

y -= 10

add_title("Conteo de preguntas y respuestas por speaker:")
preguntas = df.groupby("speaker")["es_pregunta"].sum().astype(int)
respuestas = df.groupby("speaker")["es_respuesta"].sum().astype(int)
for speaker in preguntas.index:
    add_text(f"{speaker}: preguntas={preguntas[speaker]}, respuestas={respuestas[speaker]}")

y -= 10

add_title("Distribución de emociones detectadas:")
emociones = df["emocion_detectada"].value_counts()
for emocion, cantidad in emociones.items():
    add_text(f"{emocion.capitalize()}: {cantidad}")

# === PÁGINA 3+: Gráficos ===
c.showPage()
y = height - inch

graficos = [
    ("velocidad.png", "Velocidad de habla: mide cuán rápido habla cada participante."),
    ("volumen.png", "Volumen RMS: mide la energía vocal del hablante."),
    ("pitch.png", "Pitch medio: tono promedio, útil para detectar tensión o emociones."),
    ("emociones.png", "Emociones: distribución de sentimientos detectados en los turnos."),
    ("interrupciones.png", "Interrupciones: número de turnos que superpusieron al anterior."),
    ("palabras_clave.png", "Palabras clave: términos comerciales detectados más frecuentes.")
]

for nombre, descripcion in graficos:
    ruta = f"figs/{nombre}"
    if os.path.exists(ruta):
        add_title(descripcion)
        add_image(ruta)

c.save()
print("✅ PDF generado: informe_analisis_voz.pdf")





