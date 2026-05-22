import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

data = {}
cache = {}

MEM_FILE = "memoria_usuarios.json"
memoria = {}

FILTROS_FILE = "ia_modelo.json"
filtros = {}


def cargar_json():
    global data
    with open("ia_modelo.json", "r", encoding="utf-8") as f:
        data = json.load(f)


def cargar_filtros():
    global filtros
    if os.path.exists(FILTROS_FILE):
        with open(FILTROS_FILE, "r", encoding="utf-8") as f:
            filtros = json.load(f)
    else:
        filtros = {"INAPROPIADO": [], "BLOQUEADO": []}


def cargar_memoria_archivo():
    global memoria
    if os.path.exists(MEM_FILE):
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            memoria = json.load(f)
    else:
        memoria = {}

def guardar_memoria_archivo():
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=4)

def get_user(user):
    if user not in memoria:
        memoria[user] = {
            "nombre": None,
            "historial": []
        }
    return memoria[user]


def limpiar(t):
    return t.lower().strip()


def emb(cat):
    if cat not in cache:
        cache[cat] = modelo.encode([e["input"] for e in data[cat]])
    return cache[cat]

def match(texto, cat):
    if cat not in data or len(data[cat]) == 0:
        return None

    t_emb = modelo.encode([texto])
    d_emb = emb(cat)

    sim = cosine_similarity(t_emb, d_emb)[0]
    idx = np.argmax(sim)

    if sim[idx] < 0.55:
        return None

    return data[cat][idx]["output"]


def moderar(t):
    t = limpiar(t)

    for palabra in filtros.get("INAPROPIADO", []):
        if palabra in t:
            return "INAPROPIADO"

    for palabra in filtros.get("BLOQUEADO", []):
        if palabra in t:
            return "BLOQUEADO"

    return "OK"

def emocion(t):
    t = limpiar(t)

    if any(x in t for x in ["urgente", "ya", "rapido"]):
        return "URGENTE"

    if any(x in t for x in ["molesto", "enojado", "odio"]):
        return "NEGATIVO"

    if any(x in t for x in ["gracias", "excelente", "bien"]):
        return "POSITIVO"

    return "NEUTRO"


def procesar_soporte(t, user="default"):

    if moderar(t) != "OK":
        return {
            "tipo": "SOPORTE",
            "estado": "BLOQUEADO",
            "respuesta": "No permitido"
        }

    t = limpiar(t)
    u = get_user(user)

   
    if "me llamo" in t:
        nombre = t.split("me llamo")[-1].strip().capitalize()
        u["nombre"] = nombre

        r = f"👋 Mucho gusto {nombre}, soy Hero Market IA."

        u["historial"].append({"in": t, "out": r})
        guardar_memoria_archivo()

        return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}

  
    if any(x in t for x in ["hola", "buenas", "hi", "hello"]):

        nombre = u["nombre"]

        if nombre:
            r = f"🤖 Hola {nombre}, soy Hero Market IA."
        else:
            r = "🤖 Hola, soy Hero Market IA."

        r += "\n📩 notiene@gmail.com\n📱 968085026"

        u["historial"].append({"in": t, "out": r})
        guardar_memoria_archivo()

        return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}

   
    if "cómo me llamo" in t or "como me llamo" in t:
        nombre = u["nombre"]

        if nombre:
            r = f"👀 Te llamas {nombre}"
        else:
            r = "No recuerdo tu nombre aún"

        return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}

   
    if "recuerdas" in t:
        hist = u["historial"]

        if len(hist) == 0:
            r = "No tengo recuerdos contigo"
        else:
            r = "Recuerdo:\n" + "\n".join(
                [f"- {h['in']} → {h['out']}" for h in hist[-5:]]
            )

        return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}

    
    r = match(t, "SOPORTE")

    if r:
        u["historial"].append({"in": t, "out": r})
        guardar_memoria_archivo()
        return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}

    r = "No tengo respuesta exacta 😕"

    u["historial"].append({"in": t, "out": r})
    guardar_memoria_archivo()

    return {"tipo": "SOPORTE", "estado": "OK", "respuesta": r}


def procesar_correo(t):

    t = limpiar(t)
    r = match(t, "CORREO")

    if r:
        return {"tipo": "CORREO", "estado": "OK", "resultado": r}

    return {"tipo": "CORREO", "estado": "OK", "resultado": "NEUTRO"}


def procesar_comentario(t):

    if moderar(t) != "OK":
        return {
            "tipo": "COMENTARIO",
            "estado": "BLOQUEADO",
            "sentimiento": "BLOQUEADO"
        }

    emo = match(t, "COMENTARIO")

    if emo:
        return {"tipo": "COMENTARIO", "estado": "OK", "sentimiento": emo}

    return {
        "tipo": "COMENTARIO",
        "estado": "OK",
        "sentimiento": emocion(t)
    }


cargar_json()
cargar_memoria_archivo()
cargar_filtros()