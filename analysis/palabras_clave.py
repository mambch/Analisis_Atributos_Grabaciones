PALABRAS_SCRIPT = [
    "accidentes personales", "ahorro", "telemedicina", "beneficios",
    "cobertura", "certificado", "poliza", "descuento", "portal", "cliente"
]

def detectar_palabras_clave(texto):
    """
    Detecta palabras clave comerciales en un texto.
    """
    texto = texto.lower()
    return [palabra for palabra in PALABRAS_SCRIPT if palabra in texto]
