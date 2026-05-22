from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import Ia as ia

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ia.cargar_json()
ia.cargar_memoria_archivo()

class Entrada(BaseModel):
    texto: str
    modo: str
    user: str = "default"

@app.get("/")
def home():
    return {"mensaje": "IA funcionando correctamente"}

def ejecutar(func, *args):
    try:
        return func(*args)
    except Exception as e:
        return {"error": str(e)}

@app.post("/ia")
def main(data: Entrada):

    modo = data.modo.lower()

    if modo == "soporte":
        return ejecutar(ia.procesar_soporte, data.texto, data.user)

    if modo == "correo":
        return ejecutar(ia.procesar_correo, data.texto)

    if modo == "comentario":
        return ejecutar(ia.procesar_comentario, data.texto)

    return {"error": "Modo inválido"}