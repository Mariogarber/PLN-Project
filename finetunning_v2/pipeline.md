# Pipeline de Finetuning de mT5 para Detoxificación Multilingüe

## 1. Objetivo del Proyecto
El objetivo es entrenar un modelo **mT5** para realizar **detoxificación de texto multilingüe**, es decir:

> **Transformar una frase tóxica en una versión no tóxica**, manteniendo en lo posible el contenido original y permitiendo una reescritura parcial controlada.

El modelo debe funcionar de forma competente al menos en **6 idiomas principales** del dataset.

## 2. Descripción del Dataset

- **Número de pares tóxico → neutro:** ~3600  
- **Idiomas presentes:** `en`, `am`, `ar`, `de`, `es`, `hi`, `ru`, `uk`, `zh`  
- **Longitud típica de oración:** 10–40 tokens  
- **Características del dataset:**  
  - Gran variabilidad en el grado de reescritura neutral.  
  - Mezcla de pares con alta y baja similitud.  
  - Dominio desconocido.

## 3. Problemas Identificados
1. Dataset pequeño → riesgo de sobreajuste.  
2. Multilingüe con pocos ejemplos por idioma.  
3. Variabilidad en la cantidad de reescritura.  
4. Necesidad de preservar palabras sin mecanismo explícito.

## 4. Formulación de la Tarea

### Estilo de input-output
```
<lang> detoxify_keep_meaning: <frase_tóxica>
```

Ejemplo:
```
<es> detoxify_keep_meaning: Eres un idiota que no entiende nada.
```

Output:
```
Estás actuando de forma poco razonable y no estás entendiendo la situación.
```

## 5. Preparación del Dataset

### 5.1 División Estratificada
El split debe respetar:
- idioma,
- longitud,
- nivel de solapamiento.

### 5.2 Curricular Learning
- **Fase 1:** pares con solapamiento > 70%  
- **Fase 2:** todos los pares

## 6. Elección del Modelo y Técnica PEFT

### 6.1 Modelo base
- Recomendado: **mT5-base**

### 6.2 Técnicas PEFT
1. Prefix-Tuning / P-Tuning v2  
2. LoRA ligera en decoder  
3. Encoder congelado  

## 7. Entrenamiento

### Configuraciones clave
- Loss: cross-entropy  
- LR pequeña  
- Decoding con beam search  

### Pérdidas opcionales
```
Loss = CE + λ1 * (1 - BERTScore) + λ2 * ToxicityPenalty(output)
```

## 8. Evaluación

### Métricas automáticas
- BERTScore  
- Overlap léxico  
- Toxicidad del output  
- BLEURT / SentenceMoverSimilarity  

### Evaluación manual
- Por idioma  
- Casos adversariales  

## 9. Inferencia y Decoding
- Beam Search (3–4)  
- Top-p opcional  
- Length penalty para controlar reescritura  

## 10. Riesgos y Mitigación

| Riesgo | Mitigación |
|--------|------------|
| Sobre-reescritura | Curricular learning + λ1 |
| Toxicidad residual | λ2 + decoding conservador |
| Degradación multilingüe | Encoder congelado |
| Alucinaciones | Penalización de repetición |

## 11. Siguientes Pasos
1. Decidir uso de `<lang>`.  
2. Crear splits estratificados.  
3. Implementar dataset HF.  
4. Configurar PEFT.  
5. Implementar métricas.  
6. Ajustar hiperparámetros.  
7. Evaluar outputs finales.  
