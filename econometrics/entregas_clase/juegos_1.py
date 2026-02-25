# Definimos las estrategias posibles
estrategias = ["Contratar", "No Contratar"]

# Matriz de pagos: (Beneficio Tuyo, Beneficio Rival) en millones
# El formato del diccionario es: (Tu_Estrategia, Estrategia_Rival): (Tus_Ganancias, Ganancias_Rival)
matriz_pagos = {
    ("Contratar", "Contratar"): (4, 4),
    ("Contratar", "No Contratar"): (20, 1),
    ("No Contratar", "Contratar"): (1, 20),
    ("No Contratar", "No Contratar"): (10, 10)
}

print("=== ANÁLISIS DE TEORÍA DE JUEGOS ===")
print("Objetivo: Encontrar la estrategia dominante para maximizar beneficios.\n")

estrategias_optimas = []

# Analizamos qué te conviene hacer ante CADA posible decisión del rival
for decision_rival in estrategias:
    print(f"Supongamos que el rival decide: {decision_rival.upper()}")
    
    mejor_ganancia = -1
    mejor_decision = ""
    
    # Evaluamos tus opciones
    for tu_decision in estrategias:
        tus_ganancias, ganancias_rival = matriz_pagos[(tu_decision, decision_rival)]
        print(f"  - Si tú decides '{tu_decision}', ganas {tus_ganancias} millones.")
        
        # Guardamos cuál es la opción que te da más dinero
        if tus_ganancias > mejor_ganancia:
            mejor_ganancia = tus_ganancias
            mejor_decision = tu_decision
            
    print(f"  >>> CONCLUSIÓN PARCIAL: Te conviene '{mejor_decision}' (ganas {mejor_ganancia} millones)\n")
    estrategias_optimas.append(mejor_decision)

# Verificamos si hay una estrategia dominante (si la mejor decisión es siempre la misma)
print("=== RESULTADO FINAL ===")
if estrategias_optimas.count(estrategias_optimas[0]) == len(estrategias_optimas):
    estrategia_dominante = estrategias_optimas[0]
    print(f"Tu decisión para maximizar beneficios es: {estrategia_dominante.upper()}")
    print("Explicación: Es tu ESTRATEGIA DOMINANTE porque te da el mayor beneficio sin importar lo que haga el rival.")
else:
    print("No tienes una estrategia dominante pura.")