from fastapi import FastAPI
from pydantic import BaseModel

# 👇 IMPORTAS TODO TU SISTEMA IA
from Ia import (
    cargar,
    procesar_soporte,
    procesar_correo,
    procesar_busqueda,
    procesar_comentario
)

# =========================
# INICIAR API
# =========================
app = FastAPI()

# cargar modelo al iniciar
cargar()

# =========================
# MODELO DE ENTRADA
# =========================
class Entrada(BaseModel):
    texto: str
    modo: str  # soporte, correo, busqueda, comentario

# =========================
# ENDPOINT PRINCIPAL
# =========================
@app.post("/ia")
def usar_api(data_input: Entrada):
    texto = data_input.texto
    modo = data_input.modo.lower()

    if modo == "soporte":
        respuesta = procesar_soporte(texto)
    elif modo == "correo":
        respuesta = procesar_correo(texto)
    elif modo == "busqueda":
        respuesta = procesar_busqueda(texto)
    elif modo == "comentario":
        respuesta = procesar_comentario(texto)
    else:
        return {"error": "Modo inválido"}

    return {
        "respuesta": respuesta
    }

# =========================
# TEST RÁPIDO
# =========================
@app.get("/")
def home():
    return {"mensaje": "API IA funcionando 🚀"}