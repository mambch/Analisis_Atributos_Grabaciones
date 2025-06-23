def detectar_interrupcion(seg_actual, seg_anterior):
    """
    Detecta si un segmento interrumpe al anterior (superposición temporal).
    """
    if not seg_anterior:
        return False
    return seg_actual['start'] < seg_anterior['end']

