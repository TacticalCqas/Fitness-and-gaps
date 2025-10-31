import random
import pandas as pd
import blosum
import copy
import time
import numpy as np
from itertools import combinations
from typing import List, Tuple
import matplotlib.pyplot as plt


# Auxiliares para las gráficas

def promedio_gaps_poblacion(poblacion):
    """Promedio de gaps por individuo en la población."""
    if not poblacion:
        return 0.0
    total_gaps = sum(seq.count('-') for ind in poblacion for seq in ind)
    return total_gaps / len(poblacion)

def crear_grafica_unica_completa(original, mejorado):
    """Crea una gráfica que muestra solo fitness y gaps de ambos algoritmos con marcadores finales"""
    
    # Configurar la figura
    plt.figure(figsize=(15, 8))
    
    generaciones_original = range(len(original.historial_mejor_fitness))
    generaciones_mejorado = range(len(mejorado.historial_mejor_fitness))
    
    # ===== CALCULAR VALORES FINALES =====
    mejor_original = original.historial_mejor_fitness[-1]
    mejor_mejorado = mejorado.historial_mejor_fitness[-1]
    gaps_final_original = original.historial_promedio_gaps[-1] if original.historial_promedio_gaps else 0
    gaps_final_mejorado = mejorado.historial_promedio_gaps[-1] if mejorado.historial_promedio_gaps else 0
    
    # ===== PARTE 1: COMPARACIÓN PRINCIPAL DE FITNESS =====
    
    # Líneas principales de fitness
    line_mejor_mejorado, = plt.plot(generaciones_mejorado, mejorado.historial_mejor_fitness, 
             'b-', linewidth=3, label='Mejorado - Mejor Fitness', alpha=0.9)
    
    line_mejor_original, = plt.plot(generaciones_original, original.historial_mejor_fitness, 
             'r-', linewidth=3, label='Original - Mejor Fitness', alpha=0.9)
    
    line_prom_mejorado, = plt.plot(generaciones_mejorado, mejorado.historial_promedio_fitness, 
             'b--', linewidth=2, label='Mejorado - Promedio Fitness', alpha=0.7)
    
    line_prom_original, = plt.plot(generaciones_original, original.historial_promedio_fitness, 
             'r--', linewidth=2, label='Original - Promedio Fitness', alpha=0.7)
    
    # ===== PARTE 2: LÍNEAS DE GAPS (en ejes secundarios) =====
    
    # Crear eje secundario para gaps
    ax2 = plt.gca().twinx()
    
    # Líneas de gaps
    line_gaps_original, = ax2.plot(generaciones_original, original.historial_promedio_gaps, 
             'orange', linewidth=2, linestyle='-', marker='o', markersize=3,
             label='Original - Gaps', alpha=0.8)
    
    line_gaps_mejorado, = ax2.plot(generaciones_mejorado, mejorado.historial_promedio_gaps, 
             'green', linewidth=2, linestyle='-', marker='s', markersize=3,
             label='Mejorado - Gaps', alpha=0.8)
    
    # ===== AÑADIR MARCADORES DE VALORES FINALES =====
    
    # Marcador para fitness final original
    plt.annotate(f'Fitness: {mejor_original:.0f}', 
                xy=(generaciones_original[-1], mejor_original),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # Marcador para fitness final mejorado
    plt.annotate(f'Fitness: {mejor_mejorado:.0f}', 
                xy=(generaciones_mejorado[-1], mejor_mejorado),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, color='blue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    # Marcador para gaps final original
    ax2.annotate(f'Gaps: {gaps_final_original:.1f}', 
                xy=(generaciones_original[-1], gaps_final_original),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=10, color='orange', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
    
    # Marcador para gaps final mejorado
    ax2.annotate(f'Gaps: {gaps_final_mejorado:.1f}', 
                xy=(generaciones_mejorado[-1], gaps_final_mejorado),
                xytext=(-60, -20), textcoords='offset points',
                fontsize=10, color='green', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    # Configurar eje secundario (gaps)
    ax2.set_ylabel('Promedio de Gaps', fontsize=12, fontweight='bold', color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax2.grid(False)
    
    # ===== CONFIGURACIÓN PRINCIPAL =====
    
    plt.xlabel('Generación', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness', fontsize=12, fontweight='bold', color='darkblue')
    plt.gca().tick_params(axis='y', labelcolor='darkblue')
    
    # Calcular mejora porcentual
    mejora_porcentual = ((mejor_mejorado - mejor_original) / abs(mejor_original)) * 100
    
    plt.title(f'Comparación: Fitness y Gaps - Mejora: +{mejora_porcentual:.1f}%', 
              fontsize=16, fontweight='bold', pad=20)
    
    # ===== COMBINAR LEYENDAS DE AMBOS EJES =====
    
    # Obtener líneas y etiquetas de ambos ejes
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Crear leyenda unificada
    plt.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.15),
               fontsize=10, 
               ncol=3,
               framealpha=0.9)
    
    plt.grid(True, alpha=0.3)
    
    # ===== MOSTRAR RESULTADOS EN CONSOLA =====
    
    promedio_original = np.mean(original.historial_promedio_fitness)
    promedio_mejorado = np.mean(mejorado.historial_promedio_fitness)
    
    # Mostrar resultados en consola
    print("\n" + "="*50)
    print("RESULTADOS COMPARATIVOS")
    print("="*50)
    
    print("\nFITNESS:")
    print(f"ORIGINAL:")
    print(f"  • Mejor: {mejor_original:.0f}")
    print(f"  • Promedio: {promedio_original:.0f}")
    
    print(f"MEJORADO:")
    print(f"  • Mejor: {mejor_mejorado:.0f}")
    print(f"  • Promedio: {promedio_mejorado:.0f}")
    print(f"  • Mejora: +{mejora_porcentual:.1f}%")
    
    print("\nGAPS:")
    print(f"ORIGINAL:")
    print(f"  • Gaps final: {gaps_final_original:.1f}")
    print(f"  • Gaps promedio: {np.mean(original.historial_promedio_gaps):.1f}")
    
    print(f"MEJORADO:")
    print(f"  • Gaps final: {gaps_final_mejorado:.1f}")
    print(f"  • Gaps promedio: {np.mean(mejorado.historial_promedio_gaps):.1f}")
    print(f"  • Diferencia: {gaps_final_mejorado - gaps_final_original:+.1f}")
    
    # ===== FINALIZAR =====
    
    plt.tight_layout()
    plt.savefig('COMPARACION_CON_MARCADORES.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfica con marcadores guardada como: 'COMPARACION_CON_MARCADORES.png'")
    plt.show()

# ==============================
# CLASES DE ALGORITMOS (sin cambios - se mantienen igual que antes)
# ==============================

class AlineamientoMultipleOriginal:
    """Implementación del algoritmo original para comparación"""
    
    def __init__(self, gap_penalty=-4):
        self.blosum62 = blosum.BLOSUM(62)
        self.gap_penalty = gap_penalty
        self.NFE = 0
        self.start_time = time.time()
        self.historial_mejor_fitness = []
        self.historial_promedio_fitness = []
        self.historial_peor_fitness = []
        self.historial_promedio_gaps = []
        
    def get_sequences(self):
        seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
        seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
        seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
        return [list(seq1), list(seq2), list(seq3)]

    def crear_individuo(self):
        return self.get_sequences()

    def crear_poblacion_inicial(self, n=10):
        individuo_base = self.crear_individuo()
        poblacion = [ [row[:] for row in individuo_base] for _ in range(n) ]
        return poblacion

    def mutar_poblacion_v2(self, poblacion, num_gaps=1):
        poblacion_mutada = []
        for individuo in poblacion:
            nuevo_individuo = []
            for fila in individuo:
                fila_mutada = fila[:]
                posiciones = set()
                for _ in range(num_gaps):
                    pos = random.randint(0, len(fila_mutada))
                    while pos in posiciones:
                        pos = random.randint(0, len(fila_mutada))
                    posiciones.add(pos)
                    fila_mutada.insert(pos, '-')
                nuevo_individuo.append(fila_mutada)
            poblacion_mutada.append(nuevo_individuo)
        return poblacion_mutada

    def igualar_longitud_secuencias(self, individuo, gap='-'):
        max_len = max(len(fila) for fila in individuo)
        individuo_igualado = [fila + [gap]*(max_len - len(fila)) for fila in individuo]
        return individuo_igualado

    def evaluar_individuo_blosum62(self, individuo):
        self.NFE += 1
        score = 0
        n_seqs = len(individuo)
        seq_len = len(individuo[0])
        for col in range(seq_len):
            for i in range(n_seqs):
                for j in range(i+1, n_seqs):
                    a = individuo[i][col]
                    b = individuo[j][col]
                    if a == '-' or b == '-':
                        score -= 4
                    else:
                        score += self.blosum62[a][b]
        return score

    def eliminar_peores(self, poblacion, scores, porcentaje=0.5):
        idx_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        n_seleccionados = int(len(poblacion) * porcentaje)
        
        ind_seleccionados = [poblacion[i] for i in idx_ordenados[:n_seleccionados]]
        scores_seleccionados = [scores[i] for i in idx_ordenados[:n_seleccionados]]
        
        return ind_seleccionados, scores_seleccionados

    def cruzar_individuos_doble_punto(self, ind1, ind2):
        hijo1 = []
        hijo2 = []
        for seq1, seq2 in zip(ind1, ind2):
            aa_indices = [i for i, a in enumerate(seq1) if a != '-']
            if len(aa_indices) < 6:
                hijo1.append(seq1[:])
                hijo2.append(seq2[:])
                continue

            intentos = 0
            while True:
                p1, p2 = sorted(random.sample(aa_indices, 2))
                if p2 - p1 >= 5 or intentos > 10:
                    break
                intentos += 1

            def cruza(seqA, seqB):
                aaA = [a for a in seqA if a != '-']
                aaB = [a for a in seqB if a != '-']
                nueva = aaA[:p1] + aaB[p1:p2] + aaA[p2:]
                resultado = []
                idx = 0
                for a in seqA:
                    if a == '-':
                        resultado.append('-')
                    else:
                        resultado.append(nueva[idx])
                        idx += 1
                return resultado

            nueva_seq1 = cruza(seq1, seq2)
            nueva_seq2 = cruza(seq2, seq1)

            hijo1.append(nueva_seq1)
            hijo2.append(nueva_seq2)

        # Mutación simple
        def mutar_simple(ind):
            nuevo = []
            for sec in ind:
                if random.random() < 0.8:
                    sec = sec[:]
                    pos = random.randint(0, len(sec))
                    sec.insert(pos, '-')
                nuevo.append(sec)
            return nuevo

        hijo1 = mutar_simple(hijo1)
        hijo2 = mutar_simple(hijo2)
        return hijo1, hijo2

    def cruzar_poblacion_doble_punto(self, poblacion):
        nueva_poblacion = []
        n = len(poblacion)
        indices = list(range(n))
        random.shuffle(indices)
        parejas = [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]
        if n % 2 == 1:
            parejas.append((indices[-1], indices[0]))
        for idx1, idx2 in parejas:
            padre1 = poblacion[idx1]
            padre2 = poblacion[idx2]
            hijo1, hijo2 = self.cruzar_individuos_doble_punto(padre1, padre2)
            nueva_poblacion.append(copy.deepcopy(padre1))
            nueva_poblacion.append(copy.deepcopy(padre2))
            nueva_poblacion.append(hijo1)
            nueva_poblacion.append(hijo2)
        return nueva_poblacion[:2*n]

    def validar_integridad(self, poblacion, originales):
        for individuo in poblacion:
            for seq, seq_orig in zip(individuo, originales):
                seq_sin_gaps = [a for a in seq if a != '-']
                if seq_sin_gaps != seq_orig:
                    return False
        return True

    def ejecutar(self, generaciones=100):
        veryBest = None
        fitnessVeryBest = None
        poblacion = self.crear_poblacion_inicial(10)
        poblacion = self.mutar_poblacion_v2(poblacion, num_gaps=1)
        poblacion = [self.igualar_longitud_secuencias(ind) for ind in poblacion]
        scores = [self.evaluar_individuo_blosum62(ind) for ind in poblacion]
        poblacion, scores = self.eliminar_peores(poblacion, scores)

        # Registrar punto inicial
        self.historial_mejor_fitness.append(max(scores))
        self.historial_promedio_fitness.append(np.mean(scores))
        self.historial_peor_fitness.append(min(scores))
        self.historial_promedio_gaps.append(promedio_gaps_poblacion(poblacion))
        
        for gen in range(generaciones):
            poblacion = self.cruzar_poblacion_doble_punto(poblacion)
            poblacion = [self.igualar_longitud_secuencias(ind) for ind in poblacion]
            scores = [self.evaluar_individuo_blosum62(ind) for ind in poblacion]
            poblacion, scores = self.eliminar_peores(poblacion, scores)
            
            best_fitness = max(scores)
            avg_fitness = np.mean(scores)
            worst_fitness = min(scores)
            
            if veryBest is None or best_fitness > fitnessVeryBest:
                fitnessVeryBest = best_fitness
            
            self.historial_mejor_fitness.append(fitnessVeryBest)
            self.historial_promedio_fitness.append(avg_fitness)
            self.historial_peor_fitness.append(worst_fitness)
            self.historial_promedio_gaps.append(promedio_gaps_poblacion(poblacion))
            
        return fitnessVeryBest, self.historial_mejor_fitness, self.historial_promedio_fitness


class AlineamientoMultipleAvanzado:
    """Implementación del algoritmo mejorado"""
    
    def __init__(self, gap_penalty=-4, poblacion_size=50, generaciones=200, 
                 prob_mutacion=0.3, prob_cruce=0.8, elite_size=5):
        self.blosum62 = blosum.BLOSUM(62)
        self.gap_penalty = gap_penalty
        self.poblacion_size = poblacion_size
        self.generaciones = generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.elite_size = elite_size
        
        self.NFE = 0
        self.start_time = time.time()
        self.veryBest = None
        self.fitnessVeryBest = -float('inf')
        
        # Para gráficas y análisis
        self.historial_fitness = []
        self.historial_mejor_fitness = []
        self.historial_promedio_fitness = []
        self.historial_peor_fitness = []
        self.historial_diversidad = []
        self.historial_nfe = []
        self.historial_tiempo = []
        self.mejores_por_generacion = []
        self.historial_promedio_gaps = []
        
        # Secuencias originales
        self.secuencias_originales = self._get_sequences()

    def _get_sequences(self) -> List[List[str]]:
        """Retorna las secuencias originales como listas de caracteres."""
        seq1 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRDLYDDDDKDRWGKLVVLGAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV"
        seq2 = "MKTLLVAAAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQKELQKQLGQKAKEL"
        seq3 = "MAVTQGQKLVVLGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFAVVAGGQGQAEKLVKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKAKELQKQLEQKALCVFAIN"
        return [list(seq) for seq in [seq1, seq2, seq3]]

    def crear_individuo_aleatorio(self) -> List[List[str]]:
        """Crea un individuo con inserción aleatoria de gaps."""
        individuo = []
        for secuencia in self.secuencias_originales:
            sec = secuencia[:]
            # Insertar entre 1 y 5 gaps aleatorios
            num_gaps = random.randint(1, 5)
            for _ in range(num_gaps):
                pos = random.randint(0, len(sec))
                sec.insert(pos, '-')
            individuo.append(sec)
        return individuo

    def crear_poblacion_inicial(self) -> List[List[List[str]]]:
        """Crea la población inicial diversa."""
        return [self.crear_individuo_aleatorio() for _ in range(self.poblacion_size)]

    def calcular_diversidad(self, poblacion: List[List[List[str]]]) -> float:
        """Calcula la diversidad de la población basada en las longitudes y posiciones de gaps."""
        if len(poblacion) <= 1:
            return 0.0
        
        diversidades = []
        for i in range(len(poblacion)):
            for j in range(i + 1, len(poblacion)):
                diff = 0
                total = 0
                for seq1, seq2 in zip(poblacion[i], poblacion[j]):
                    # Comparar longitudes
                    diff += abs(len(seq1) - len(seq2))
                    # Comparar patrones de gaps
                    gaps1 = [idx for idx, char in enumerate(seq1) if char == '-']
                    gaps2 = [idx for idx, char in enumerate(seq2) if char == '-']
                    diff += len(set(gaps1) ^ set(gaps2))  # Diferencia simétrica
                    total += max(len(seq1), len(seq2))
                if total > 0:
                    diversidades.append(diff / total)
        
        return np.mean(diversidades) if diversidades else 0.0

    def mutacion_insercion_gap(self, individuo: List[List[str]]) -> List[List[str]]:
        """Inserta un gap aleatorio en una secuencia aleatoria."""
        nuevo_individuo = [seq[:] for seq in individuo]
        seq_idx = random.randint(0, len(nuevo_individuo) - 1)
        pos = random.randint(0, len(nuevo_individuo[seq_idx]))
        nuevo_individuo[seq_idx].insert(pos, '-')
        return nuevo_individuo

    def mutacion_eliminacion_gap(self, individuo: List[List[str]]) -> List[List[str]]:
        """Elimina un gap aleatorio si existe."""
        nuevo_individuo = [seq[:] for seq in individuo]
        # Encontrar secuencias que tengan gaps
        secuencias_con_gaps = [i for i, seq in enumerate(nuevo_individuo) 
                              if any(char == '-' for char in seq)]
        
        if secuencias_con_gaps:
            seq_idx = random.choice(secuencias_con_gaps)
            gaps_positions = [i for i, char in enumerate(nuevo_individuo[seq_idx]) 
                            if char == '-']
            if gaps_positions:
                pos = random.choice(gaps_positions)
                del nuevo_individuo[seq_idx][pos]
        
        return nuevo_individuo

    def mutacion_desplazamiento_gap(self, individuo: List[List[str]]) -> List[List[str]]:
        """Desplaza un gap a una posición adyacente."""
        nuevo_individuo = [seq[:] for seq in individuo]
        secuencias_con_gaps = [i for i, seq in enumerate(nuevo_individuo) 
                              if any(char == '-' for char in seq)]
        
        if secuencias_con_gaps:
            seq_idx = random.choice(secuencias_con_gaps)
            gaps_positions = [i for i, char in enumerate(nuevo_individuo[seq_idx]) 
                            if char == '-']
            
            if gaps_positions:
                gap_pos = random.choice(gaps_positions)
                # Elegir dirección del desplazamiento
                direction = random.choice([-1, 1])
                new_pos = gap_pos + direction
                
                # Verificar que la nueva posición es válida
                if 0 <= new_pos < len(nuevo_individuo[seq_idx]):
                    # Mover el gap
                    nuevo_individuo[seq_idx][gap_pos] = nuevo_individuo[seq_idx][new_pos]
                    nuevo_individuo[seq_idx][new_pos] = '-'
        
        return nuevo_individuo

    def mutar_individuo(self, individuo: List[List[str]]) -> List[List[str]]:
        """Aplica una mutación aleatoria al individuo."""
        operadores_mutacion = [
            self.mutacion_insercion_gap,
            self.mutacion_eliminacion_gap,
            self.mutacion_desplazamiento_gap
        ]
        
        # Aplicar múltiples mutaciones con probabilidad decreciente
        individuo_mutado = [seq[:] for seq in individuo]
        for _ in range(3):  # Máximo 3 mutaciones
            if random.random() < self.prob_mutacion:
                mutacion = random.choice(operadores_mutacion)
                individuo_mutado = mutacion(individuo_mutado)
        
        return individuo_mutado

    def mutar_poblacion(self, poblacion: List[List[List[str]]]) -> List[List[List[str]]]:
        """Aplica mutación a toda la población (excepto a la élite)."""
        return [self.mutar_individuo(ind) if random.random() < self.prob_mutacion else ind 
                for ind in poblacion]

    def igualar_longitud_secuencias(self, individuo: List[List[str]], gap: str = '-') -> List[List[str]]:
        """Equaliza la longitud de las secuencias añadiendo gaps al final."""
        max_len = max(len(fila) for fila in individuo)
        return [fila + [gap] * (max_len - len(fila)) for fila in individuo]

    def evaluar_individuo_blosum62(self, individuo: List[List[str]]) -> int:
        """Evalúa el fitness de un individuo usando BLOSUM62."""
        self.NFE += 1
        individuo_igualado = self.igualar_longitud_secuencias(individuo)
        score = 0
        n_seqs = len(individuo_igualado)
        seq_len = len(individuo_igualado[0])
        
        # Pre-calcular todas las combinaciones de pares de secuencias
        pares = list(combinations(range(n_seqs), 2))
        
        for col in range(seq_len):
            for i, j in pares:
                a = individuo_igualado[i][col]
                b = individuo_igualado[j][col]
                if a == '-' or b == '-':
                    score += self.gap_penalty
                else:
                    score += self.blosum62[a][b]
        return score

    def seleccion_por_torneo(self, poblacion: List[List[List[str]]], scores: List[int], 
                           k: int = 3) -> List[List[List[str]]]:
        """Selección por torneo de tamaño k."""
        seleccionados = []
        for _ in range(len(poblacion)):
            indices = random.sample(range(len(poblacion)), k)
            mejor_idx = max(indices, key=lambda i: scores[i])
            seleccionados.append(copy.deepcopy(poblacion[mejor_idx]))
        return seleccionados

    def cruce_un_punto(self, padre1: List[List[str]], padre2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Cruza en un punto por cada secuencia."""
        hijo1, hijo2 = [], []
        
        for seq1, seq2 in zip(padre1, padre2):
            if len(seq1) < 2 or len(seq2) < 2:
                hijo1.append(seq1[:])
                hijo2.append(seq2[:])
                continue
            
            punto = random.randint(1, min(len(seq1), len(seq2)) - 1)
            hijo1_seq = seq1[:punto] + seq2[punto:]
            hijo2_seq = seq2[:punto] + seq1[punto:]
            
            hijo1.append(hijo1_seq)
            hijo2.append(hijo2_seq)
        
        return hijo1, hijo2

    def cruce_dos_puntos(self, padre1: List[List[str]], padre2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Cruza en dos puntos por cada secuencia."""
        hijo1, hijo2 = [], []
        
        for seq1, seq2 in zip(padre1, padre2):
            if len(seq1) < 3 or len(seq2) < 3:
                hijo1.append(seq1[:])
                hijo2.append(seq2[:])
                continue
            
            puntos = sorted(random.sample(range(1, min(len(seq1), len(seq2))), 2))
            p1, p2 = puntos
            
            hijo1_seq = seq1[:p1] + seq2[p1:p2] + seq1[p2:]
            hijo2_seq = seq2[:p1] + seq1[p1:p2] + seq2[p2:]
            
            hijo1.append(hijo1_seq)
            hijo2.append(hijo2_seq)
        
        return hijo1, hijo2

    def cruce_uniforme(self, padre1: List[List[str]], padre2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Cruza uniforme por cada posición."""
        hijo1, hijo2 = [], []
        
        for seq1, seq2 in zip(padre1, padre2):
            hijo1_seq = []
            hijo2_seq = []
            min_len = min(len(seq1), len(seq2))
            
            for i in range(min_len):
                if random.random() < 0.5:
                    hijo1_seq.append(seq1[i])
                    hijo2_seq.append(seq2[i])
                else:
                    hijo1_seq.append(seq2[i])
                    hijo2_seq.append(seq1[i])
            
            if len(seq1) > min_len:
                hijo1_seq.extend(seq1[min_len:])
                hijo2_seq.extend(seq1[min_len:])
            elif len(seq2) > min_len:
                hijo1_seq.extend(seq2[min_len:])
                hijo2_seq.extend(seq2[min_len:])
            
            hijo1.append(hijo1_seq)
            hijo2.append(hijo2_seq)
        
        return hijo1, hijo2

    def cruzar_poblacion(self, poblacion: List[List[List[str]]], scores: List[int]) -> List[List[List[str]]]:
        """Realiza cruce entre individuos de la población usando múltiples operadores."""
        nueva_poblacion = []
        
        # Preservar élite
        indices_ordenados = np.argsort(scores)[::-1]
        elite = [copy.deepcopy(poblacion[i]) for i in indices_ordenados[:self.elite_size]]
        nueva_poblacion.extend(elite)
        
        # Selección por torneo para padres
        padres = self.seleccion_por_torneo(poblacion, scores)
        
        # Cruzar para generar el resto de la población
        operadores_cruce = [self.cruce_un_punto, self.cruce_dos_puntos, self.cruce_uniforme]
        
        while len(nueva_poblacion) < self.poblacion_size:
            if random.random() < self.prob_cruce:
                padre1, padre2 = random.sample(padres, 2)
                operador = random.choice(operadores_cruce)
                hijo1, hijo2 = operador(padre1, padre2)
                
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.poblacion_size:
                    nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(random.choice(padres))
        
        return nueva_poblacion[:self.poblacion_size]

    def validar_integridad(self, poblacion: List[List[List[str]]]) -> bool:
        """Valida que al remover gaps se obtengan las secuencias originales."""
        for individuo in poblacion:
            for seq_alineada, seq_original in zip(individuo, self.secuencias_originales):
                seq_sin_gaps = [a for a in seq_alineada if a != '-']
                if seq_sin_gaps != seq_original:
                    return False
        return True

    def obtener_mejor(self, poblacion: List[List[List[str]]], scores: List[int]) -> Tuple[List[List[str]], int]:
        """Retorna el mejor individuo y su score."""
        idx_mejor = np.argmax(scores)
        return copy.deepcopy(poblacion[idx_mejor]), scores[idx_mejor]

    def _actualizar_historial(self, poblacion, scores):
        """Actualiza todas las métricas del historial."""
        self.historial_fitness.append(scores)
        self.historial_mejor_fitness.append(max(scores))
        self.historial_promedio_fitness.append(np.mean(scores))
        self.historial_peor_fitness.append(min(scores))
        self.historial_diversidad.append(self.calcular_diversidad(poblacion))
        self.historial_nfe.append(self.NFE)
        self.historial_tiempo.append(time.time() - self.start_time)
        self.historial_promedio_gaps.append(promedio_gaps_poblacion(poblacion))

    def ejecutar(self) -> Tuple[List[List[str]], int]:
        """Ejecuta el algoritmo evolutivo completo."""
        # Inicialización
        poblacion = self.crear_poblacion_inicial()
        
        # Evaluación inicial
        scores = [self.evaluar_individuo_blosum62(ind) for ind in poblacion]
        
        # Inicializar historial
        self._actualizar_historial(poblacion, scores)
        
        # Evolución
        for generacion in range(self.generaciones):
            poblacion = self.cruzar_poblacion(poblacion, scores)
            poblacion = self.mutar_poblacion(poblacion)
            scores = [self.evaluar_individuo_blosum62(ind) for ind in poblacion]
            
            best_actual, fitness_best = self.obtener_mejor(poblacion, scores)
            if fitness_best > self.fitnessVeryBest:
                self.veryBest = best_actual
                self.fitnessVeryBest = fitness_best
                self.mejores_por_generacion.append((generacion, fitness_best))
            
            self._actualizar_historial(poblacion, scores)
        
        return self.veryBest, self.fitnessVeryBest


def ejecutar_comparacion():
    """Ejecuta ambos algoritmos y genera la gráfica simple"""
    print("Ejecutando comparación de algoritmos...")
    
    # Ejecutar algoritmo original
    print("1. Algoritmo Original...")
    original = AlineamientoMultipleOriginal()
    fitness_original, historial_mejor_original, historial_promedio_original = original.ejecutar(generaciones=100)
    
    # Ejecutar algoritmo mejorado
    print("2. Algoritmo Mejorado...")
    mejorado = AlineamientoMultipleAvanzado(
        poblacion_size=50,
        generaciones=100,
        prob_mutacion=0.3,
        prob_cruce=0.8,
        elite_size=5
    )
    mejor_solucion, fitness_mejorado = mejorado.ejecutar()
    
    # Validar integridad
    validez_original = original.validar_integridad([original.crear_individuo()], mejorado.secuencias_originales)
    validez_mejorado = mejorado.validar_integridad([mejor_solucion])
    
    print(f"✓ Validación Original: {validez_original}")
    print(f"✓ Validación Mejorado: {validez_mejorado}")
    
    # Crear gráfica simple
    print("3. Generando gráfica con marcadores...")
    crear_grafica_unica_completa(original, mejorado)
    
    return original, mejorado, mejor_solucion


if __name__ == "__main__":
    # Ejecutar comparación completa
    original, mejorado, mejor_solucion = ejecutar_comparacion()
