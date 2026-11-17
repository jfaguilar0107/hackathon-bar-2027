# Latent XI GT — Documento técnico · More Than A Hack 2027, Fase 1 (posesión de balón)

**Equipo:** Javier Aguilar, Elisa Samayoa, Julio Ávila
**Repositorio público:** https://github.com/jfaguilar0107/hackathon-bar-2027
**Entrega:** `Latent XI GT.csv` (199,319 filas, una etiqueta H/A/D por frame; ninguna X)

## 1. Enfoque en una página

Tratamos el problema como **segmentación**, no como clasificación frame a frame. La posesión es un estado que dura segundos, así que primero decidimos qué frames son balón muerto (D) y después, dentro de cada tramo con balón vivo, decidimos H o A con un Viterbi de dos estados que penaliza cada cambio.

Como el balón está ausente en el 41% de los frames calificados (150 huecos, mediana 10.7 s, sin huecos cortos que interpolar), la identidad del equipo con posesión se estima con dos fuentes:

1. **Semillas de control claro** (23.1% de los frames calificados): balón visible, jugador más cercano a menos de 2 m y al menos 2 m de ventaja sobre el rival.
2. **Un modelo de forma sin balón** entrenado con esas semillas como pseudo-etiquetas, que infiere el equipo con posesión desde la posición y el movimiento de ambos equipos.

## 2. Datos usados y descartados

| Dato | Uso |
|---|---|
| ATD táctico (tracking automático, 46 slots) | Usado: posiciones de jugadores y balón, velocidades calculadas desde posiciones. |
| Smart Tagging | Usado solo como señal de reinicio: inicio de Set Piece y Kick Off (Start). Probamos las fases como prior de equipo y lo descartamos (sección 8). |
| **Sportec/DFL IDSSE** (7 partidos de la Bundesliga, datos abiertos de otros partidos) | Usado para entrenar el detector de pausas (sección 4). Licencia CC-BY 4.0; Bassek et al. 2025, *Scientific Data* 12:195, doi 10.1038/s41597-025-04505-y. |
| Video, broadcast, homografía | Descartados. |
| GT público de 5 min | Solo como prueba de humo del pipeline. No se usó para entrenar, ajustar ni elegir entre versiones. |

No se etiquetó ni corrigió nada a mano, no hay constantes atadas a frames del partido salvo las ventanas calificadas oficiales (11,185–90,651 y 110,808–190,197), y no se usaron datos humanos del partido.

## 3. Reglas de posesión

- Posesión = corrida máxima de H o de A con balón vivo; D la interrumpe.
- El cambio de posesión exige evidencia **sostenida**: el costo λ por cambio del Viterbi impide cambios por un solo toque o por un duelo.
- Balón suelto o disputado: se mantiene el último poseedor (el Viterbi no cambia sin evidencia).
- Los 2,934 frames calificados del 2º tiempo anteriores al inicio real (113,742) son D.

## 4. Pipeline

1. **Semillas** (`control_claro`), descritas arriba.
2. **Modelo de forma.** XGBoost (300 árboles, profundidad 4, `random_state=0`) sobre la forma de cada equipo orientada hacia su sentido de ataque (centroide, dispersión en x e y, línea defensiva, velocidad, cambio de centroide a 1 s) más medias centradas de 5, 15 y 30 s. Peso igual por tramo de distancia al hueco más cercano. Validación por 6 bloques de tiempo: acierto 0.812 en semillas y 0.771 en el borde de huecos (< 5 s). Calibrado con *temperature scaling* (T = 1.5738, recalculado en cada corrida).
3. **Balón muerto, regla de huecos.** Un hueco del balón es pausa si el balón se perdió o reapareció a menos de 3 m de una línea, reapareció en el centro, o hay inicio de Set Piece (dentro del hueco o hasta 5 s después). También son D el balón visible a más de 1 m fuera y el tramo antes del primer Kick Off de cada tiempo. Resultado: 79 de 150 huecos, 31.2% D.
4. **Detector de pausas entrenado con DFL.** XGBoost (mismos parámetros) con estado vivo/muerto del balón como etiqueta, sobre 4 features sin balón calculadas con la misma función en DFL y ATD: velocidad mediana del conjunto de jugadores en ventanas de 3 y 10 s, mínima entre equipos en 3 s, y dispersión en y en 3 s. Solo **agrega** D a la regla (la regla conserva prioridad): P(pausa) > 0.9, se cierran huecos de menos de 2 s y se descartan tramos nuevos de menos de 5 s (el 10% de las pausas reales del DFL dura menos de 5.6 s). D total: 31.2% → 39.6% de los frames calificados.
5. **Viterbi H/A.** D es restricción dura. Dentro de cada tramo vivo, 2 estados, evidencia del modelo de forma, costo λ = 2 por cambio.

## 5. Validación sin el GT del partido

- **Detector, un partido fuera a la vez** (7 folds): AUC 0.936–0.957 por partido, 0.943 global; a umbral 0.9, precisión 0.982 y recall 0.51. Sin señales de sobreajuste (AUC 0.950 en entrenamiento contra 0.944 en datos no vistos). No se afinaron hiperparámetros.
- **Transferencia DFL → ATD:** velocidad media en pausa 1.02 m/s (DFL) y 1.07 m/s (ATD); las pausas nuevas del detector caen a 0.70 m/s contra 2.00 en el resto del juego. El detector marca 54.3% de las pausas de la regla (en DFL, 55.1% de las pausas reales).
- **Plausibilidad:** 229 posesiones, media 16.8 s, mediana 10.4 s.

## 6. Parámetros

| Parámetro | Valor | Cómo se eligió |
|---|---|---|
| Semilla: distancia / margen | 2 m / 2 m | Criterio de control claro |
| T (calibración) | 1.5738 | Minimiza log-loss OOF en semillas a < 5 s del hueco |
| Umbral del detector | 0.9 | Precisión en validación DFL fuera de partido (0.982) |
| Cierre / mínimo de pausa nueva | 2 s / 5 s | Fragmentación; mínimo anclado en el percentil 10 de pausas reales DFL |
| λ del Viterbi | 2 | Plausibilidad de conteo y duración (ver limitaciones) |

## 7. Sensibilidad de λ

| λ | posesiones | media (s) | mediana (s) | % < 3 s |
|---|---|---|---|---|
| 1 | 295 | 13.0 | 7.0 | 23.1 |
| 1.5 | 260 | 14.8 | 8.6 | 16.9 |
| **2** | **229** | **16.8** | **10.4** | **14.0** |
| 3 | 194 | 19.8 | 12.4 | 10.3 |
| 4 | 171 | 22.4 | 15.6 | 9.4 |

El acuerdo con las semillas sube al bajar λ (97.1% con λ = 1, 91.6% con λ = 4) porque un Viterbi más flexible se ajusta más a cualquier evidencia, incluido el error. Por eso no lo usamos para elegir λ.

## 8. Resultados negativos

- **Prior de Smart Tagging.** `Team` es el equipo que ejecuta la fase (en fases defensivas el poseedor es el rival) y las fases vienen en pares espejo. Como voto solo acierta 75.4% contra 81.5% del modelo, y combinado sube el acierto en semillas (0.812 → ~0.825) pero en el borde de huecos solo medio punto (0.771 → 0.776). Ganancia no concluyente: descartado.
- **Detector de D por frame con balón fuera como etiqueta:** 12.8% de D y parpadeo; descartado.
- **LightGBM** quedó por debajo de XGBoost en 3 comparaciones.
- **D como estado blando del Viterbi:** el modelo usaba H→D→A como atajo y casi no hacía cambios directos; D es restricción dura.

## 9. Limitaciones

- **No pudimos medir la mejora de spellF1** de la versión final sin etiquetas del partido. Las decisiones se tomaron con chequeos de plausibilidad y validación en otros partidos.
- **λ = 2 se eligió por plausibilidad** de conteo y duración de posesiones. Ese criterio no es del todo independiente, porque conocíamos la media pública de 14.2 s.
- **D sigue subestimado:** 39.6% de los frames calificados frente a 41–52% de balón muerto en los partidos DFL. El detector tiene recall 0.51.
- Las pausas nuevas caen más sobre frames que la versión anterior etiquetaba A (17.5%) que H (8.0%); no verificamos la causa.
- Las faltas con balón visible a menos de 3 m de jugadores pueden quedar como balón vivo.
- La velocidad de pausa en ATD y DFL no es del todo comparable (distintos proveedores y sistemas de tracking).

## 10. Reproducción

```
python -m venv .venv && .venv\Scripts\activate
pip install -r solution/requirements.txt          # Python 3.12
python solution/run.py --out "Latent XI GT.csv" --sin-cache
python evaluation/validate_submission.py "Latent XI GT.csv"
```

- `solution/run.py` lee los archivos oficiales del concurso, que no se incluyen en este repositorio. Versiones fijadas, semillas fijas y `n_jobs=1`: dos corridas dan el mismo archivo (SHA256 `F929326EEE976E3D55CC84862E2E0C58D539B92D40F510A57E02B3A353F21733`).
- `--sin-detector` regenera la versión anterior sin DFL (SHA256 `7BD3653C0E82896DDE7135E6AEBE68DA47797B12E10AC41301E20B8D6755B5AA`).
- `solution/dfl_train.npz` (200,532 filas × 4 features + etiqueta) es una tabla derivada de los 7 partidos DFL. Se obtiene con las celdas de la sección «DFL» de `notebooks/mtah_posesion.ipynb` a partir de los datos abiertos del dataset en Hugging Face (`pysport/idsse-data`).

## 11. Atribución

Datos de entrenamiento del detector: Sportec Solutions / DFL, conjunto abierto IDSSE (CC-BY 4.0). Bassek, M., et al. (2025). *Scientific Data* 12:195. doi:10.1038/s41597-025-04505-y.
