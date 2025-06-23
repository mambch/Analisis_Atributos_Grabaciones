def detectar_pregunta(texto):
    """
    Detecta si el segmento parece una pregunta.
    """
    texto = texto.strip().lower()
    if texto.endswith("?"):
        return True
    palabras_pregunta = ["cómo", "qué", "cuál", "cuándo", "dónde", "por qué", "para qué"]
    return any(p in texto for p in palabras_pregunta)

def detectar_respuesta(texto):
    """
    Detecta si el segmento parece una respuesta simple.
    """
    texto = texto.strip().lower()
    palabras_respuesta = ["sí", "no", "claro", "correcto", "perfecto", "ok", "entiendo"]
    return any(texto.startswith(p) for p in palabras_respuesta)
