import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# MODELO IA
# =========================
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# PERFIL USUARIO
# =========================
usuario = {
    "nombre": None,
    "genero": None
}

# =========================
# CARGAR / GUARDAR
# =========================
def cargar():
    global data
    try:
        with open("ia_modelo.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print("📂 Modelo cargado")
    except:
        print("⚠️ No hay modelo, creando nuevo...")
        data = {
            "SOPORTE": [],
            "CORREO": [],
            "BUSQUEDA": [],
            "TONO": [],
            "GENERO": [],
            "COMENTARIO": []
        }

def guardar():
    with open("ia_modelo.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("💾 Guardado correctamente")

# =========================
# IA
# =========================
def embed(textos):
    return modelo.encode(textos)

def similitud(texto, ejemplos):
    if not ejemplos:
        return None

    inputs = [e["input"] for e in ejemplos]
    emb_inputs = embed(inputs)
    emb_texto = embed([texto])

    sim = cosine_similarity(emb_texto, emb_inputs)[0]
    idx = sim.argmax()
    return idx

# =========================
# NORMALIZAR
# =========================
def limpiar_texto(texto):
    texto = texto.lower().strip()
    texto = texto.replace("ncecesito", "necesito")
    texto = texto.replace("necesit", "necesito")
    return texto

# =========================
# DETECTORES
# =========================
def detectar_producto(texto):
    texto = limpiar_texto(texto)

    if "router" in texto or "internet" in texto:
        return "router"
    if "switch" in texto:
        return "switch"
    if "wifi" in texto or "access point" in texto:
        return "access point"

    return None

def detectar_tono(texto):
    idx = similitud(texto, data["TONO"])
    if idx is None:
        return "normal"
    return data["TONO"][idx]["output"]

def detectar_nombre(texto):
    texto = texto.lower()

    if "me llamo" in texto:
        nombre = texto.split("me llamo")[-1].strip().split()[0]
        return nombre.capitalize()

    if "soy" in texto:
        nombre = texto.split("soy")[-1].strip().split()[0]
        return nombre.capitalize()

    return None

def detectar_genero(texto):
    texto = limpiar_texto(texto)

    if "soy hombre" in texto:
        return "masculino"
    if "soy mujer" in texto:
        return "femenino"

    idx = similitud(texto, data["GENERO"])
    if idx is None:
        return None

    return data["GENERO"][idx]["output"]

# =========================
# PERSONALIZAR
# =========================
def personalizar(respuesta, tono):
    prefijo = ""

    if usuario["nombre"]:
        if usuario["genero"] == "femenino":
            prefijo += f"{usuario['nombre']}, "
        else:
            prefijo += f"{usuario['nombre']}, "

    if tono == "urgente":
        prefijo += "lo vemos de inmediato. "
    elif tono == "molesto":
        prefijo += "lamento el inconveniente. "
    elif tono == "amable":
        prefijo += "con gusto te ayudo. "

    return prefijo + respuesta

# =========================
# RESPUESTA PRODUCTO
# =========================
def responder_producto(producto):
    if producto == "router":
        return "Te ayudo con routers 👍 ¿Lo necesitas para hogar o empresa?"
    if producto == "switch":
        return "Perfecto 👍 ¿Cuántos equipos necesitas conectar?"
    if producto == "access point":
        return "Te ayudo con WiFi 👍 ¿Es para oficina o negocio?"
    return None

# =========================
# PROCESOS
# =========================
def procesar_soporte(texto):
    texto = limpiar_texto(texto)

    # detectar nombre
    nombre = detectar_nombre(texto)
    if nombre:
        usuario["nombre"] = nombre

    # detectar genero
    genero = detectar_genero(texto)
    if genero:
        usuario["genero"] = genero

    tono = detectar_tono(texto)

    # detectar producto
    producto = detectar_producto(texto)
    if producto:
        respuesta = responder_producto(producto)
    else:
        idx = similitud(texto, data["SOPORTE"])
        if idx is None:
            respuesta = "No tengo suficiente información aún."
        else:
            base = data["SOPORTE"][idx]["output"]
            variantes = [
                base,
                "Puedes intentar: " + base,
                "Te recomiendo: " + base
            ]
            respuesta = random.choice(variantes)

    # 🔥 PRESENTACIÓN DE LA IA
    presentacion = "Hola, soy tu asistente de soporte. "

    return presentacion + personalizar(respuesta, tono)


def procesar_correo(texto):
    idx = similitud(texto, data["CORREO"])
    if idx is None:
        return "SIN DATOS"
    return data["CORREO"][idx]["output"]


def procesar_busqueda(texto):
    texto = limpiar_texto(texto)

    producto = detectar_producto(texto)
    if producto:
        return responder_producto(producto)

    idx = similitud(texto, data["BUSQUEDA"])
    if idx is None:
        return "No encontré el producto."

    producto = data["BUSQUEDA"][idx]["output"]
    return responder_producto(producto)


def procesar_comentario(texto):
    texto = limpiar_texto(texto)

    # reglas rápidas
    if any(p in texto for p in ["bueno", "excelente", "perfecto", "recomendado"]):
        return "POSITIVO"

    if any(p in texto for p in ["malo", "pésimo", "horrible", "lento"]):
        return "NEGATIVO"

    idx = similitud(texto, data["COMENTARIO"])
    if idx is None:
        return "NEUTRO"

    return data["COMENTARIO"][idx]["output"]

# =========================
# ENTRENAMIENTO
# =========================
def entrenar():
    print("\nCategorías: SOPORTE / CORREO / BUSQUEDA / TONO / GENERO / COMENTARIO")
    cat = input("Categoría: ").upper()

    if cat not in data:
        print("❌ Categoría inválida")
        return

    entrada = input("Entrada: ")
    salida = input("Salida: ")

    data[cat].append({
        "input": entrada,
        "output": salida
    })

    guardar()
    print("✅ Aprendido correctamente")

# =========================
# TEST
# =========================
def testear():
    print("\n🧪 TEST")
    modo = input("Modo (soporte/correo/busqueda/comentario): ").lower()

    while True:
        texto = input("\nInput ('salir'): ")
        if texto == "salir":
            break

        if modo == "soporte":
            print("👉", procesar_soporte(texto))
        elif modo == "correo":
            print("👉", procesar_correo(texto))
        elif modo == "busqueda":
            print("👉", procesar_busqueda(texto))
        elif modo == "comentario":
            print("👉", procesar_comentario(texto))

# =========================
# USO
# =========================
def usar():
    print("\n🤖 IA")
    modo = input("Modo (soporte/correo/busqueda/comentario): ").lower()

    while True:
        texto = input("\nTú: ")
        if texto == "salir":
            break

        if modo == "soporte":
            print("IA:", procesar_soporte(texto))
        elif modo == "correo":
            print("IA:", procesar_correo(texto))
        elif modo == "busqueda":
            print("IA:", procesar_busqueda(texto))
        elif modo == "comentario":
            print("IA:", procesar_comentario(texto))

# =========================
# MENU
# =========================
def menu():
    cargar()

    while True:
        print("\n===== MENU =====")
        print("1. Usar IA")
        print("2. Entrenar")
        print("3. Testear")
        print("4. Guardar")
        print("5. Salir")

        op = input("Opción: ")

        if op == "1":
            usar()
        elif op == "2":
            entrenar()
        elif op == "3":
            testear()
        elif op == "4":
            guardar()
        elif op == "5":
            break

if __name__ == "__main__":
    menu()