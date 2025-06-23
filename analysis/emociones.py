def detectar_emocion(texto):
    """
    Detecta emoción con heurísticas simples de palabras.
    """
    texto = texto.lower()

    if any(p in texto for p in ["gracias", "perfecto", "excelente", "bien", "tranquilo"]):
        return "positiva"
    elif any(p in texto for p in ["no entiendo", "problema", "molesto", "error"]):
        return "negativa"
    elif "..." in texto or texto.strip() == "":
        return "neutra"
    else:
        return "neutra"
