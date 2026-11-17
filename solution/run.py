"""Regenera la entrega v2 (enfoque de Javier) desde los datos del partido.

Uso:
    python solution/run.py --out evaluation/results/run_v2.csv [--sin-cache]

Necesita MTAH_DATA_DIR apuntando a la carpeta de datos (fuera del repo).
Etapas: cargar -> semillas -> orientación -> features -> modelo de forma
-> calibración -> pausas (D) -> Viterbi H/A -> CSV.
"""
import argparse
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from games import GAME, GAMES, asset_paths                    # noqa: E402
from starter.load_smart_tagging import load as cargar_tags    # noqa: E402

N_FRAMES, N_SLOTS = 199319, 46
LARGO, ANCHO = GAMES[GAME]["pitch"]
TIEMPOS = [(11185, 90652), (110808, 190198)]    # ventanas calificadas oficiales
INICIO_2T = 110808
H, A, D = 0, 1, 2
EPS = 1e-6
LAMBDA = 2                                       # costo por cambio de posesión
TRAMOS = ["0-2 s", "2-5 s", "5-10 s", ">10 s"]
DFL_TRAIN = ROOT / "solution" / "dfl_train.npz"   # features + balón muerto de 7 partidos DFL (CC-BY 4.0)
COLS_D = ["vel_3s", "vel_10s", "vel_min_3s", "sy_3s"]
UMBRAL_D = 0.9                                    # solo se marca pausa si el detector está muy seguro
CIERRE_S, MIN_PAUSA_S = 2, 5                      # rellenar huecos < 2 s; descartar pausas nuevas < 5 s
BASE = ["L_cx", "L_sx", "L_sy", "L_linea_def", "L_vel",
        "V_cx", "V_sx", "V_sy", "V_linea_def", "V_vel"]


# ---------------------------------------------------------------- utilidades
def huecos(falta, inicio, fin):
    """Inicio y largo (frames) de cada racha donde `falta` es True en [inicio, fin)."""
    x = falta[inicio:fin].astype(np.int8)
    cambios = np.diff(np.concatenate(([0], x, [0])))
    comienzos = np.flatnonzero(cambios == 1)
    finales = np.flatnonzero(cambios == -1)
    return comienzos + inicio, finales - comienzos


def modelo_xgb():
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         tree_method="hist", random_state=0, n_jobs=1)


# ---------------------------------------------------------------- 1. carga
def a_numeros(texto):
    """'x,y,vel' -> (x, y, vel). Sin dato -> NaN."""
    if texto[0] == "N":
        return (np.nan, np.nan, np.nan)
    partes = texto.split(",")
    return (float(partes[0]), float(partes[1]), float(partes[2]))


def cargar_atd(rutas, usar_cache):
    """jug (frames, 46, [x, y, vel]) y bal (frames, [x, y, vel]) en metros."""
    cache = Path(rutas["atd_raw"]).parent / "atd_arrays.npz"
    if usar_cache and cache.exists():
        d = np.load(cache)
        return d["jug"], d["bal"]
    jug = np.full((N_FRAMES, N_SLOTS, 3), np.nan, dtype=np.float32)
    bal = np.full((N_FRAMES, 3), np.nan, dtype=np.float32)
    with open(rutas["atd_raw"]) as f:
        for linea in f:
            frame, jugadores, balon = linea.strip().split(":")
            i = int(frame) - 1                       # video_frame = frame_id - 1
            for s, grupo in enumerate(jugadores.split(";")):
                jug[i, s] = a_numeros(grupo)
            bal[i] = a_numeros(balon)
    jug[:, :, 0] *= LARGO; jug[:, :, 1] *= ANCHO
    bal[:, 0] *= LARGO;    bal[:, 1] *= ANCHO
    if usar_cache:
        np.savez(cache, jug=jug, bal=bal)
    return jug, bal


def cargar_equipos(rutas):
    """Equipo de cada slot: 0 = local (primer equipo del XML), 1 = visitante."""
    raiz = ET.parse(rutas["atd_meta"]).getroot()
    equipos = [(t.get("id"), t.findtext("Name")) for t in raiz.iter("Team")]
    slots = [p.get("teamId") for p in raiz.iter("Player")]
    id_local = equipos[0][0]
    return np.array([0 if s == id_local else 1 for s in slots])


# ---------------------------------------------------------------- 2. semillas
def semillas(jug, bal, calificado, equipo_slot):
    con_balon = ~np.isnan(bal[:, 0])
    dx = jug[:, :, 0] - bal[:, [0]]
    dy = jug[:, :, 1] - bal[:, [1]]
    dist = np.nan_to_num(np.sqrt(dx**2 + dy**2), nan=np.inf)
    dist_local = np.min(np.where(equipo_slot == 0, dist, np.inf), axis=1)
    dist_visit = np.min(np.where(equipo_slot == 1, dist, np.inf), axis=1)

    vis = con_balon & calificado
    cerca = np.minimum(dist_local, dist_visit)
    with np.errstate(invalid="ignore"):                 # inf - inf sin balón -> NaN (no cuenta)
        margen = np.abs(dist_local - dist_visit)
    hay = ~np.isnan(jug[:, :, 0])
    tracking_ok = (hay[:, equipo_slot == 0].sum(axis=1) > 0) & (hay[:, equipo_slot == 1].sum(axis=1) > 0)

    control_claro = vis & (cerca < 2) & (margen >= 2) & tracking_ok
    equipo_cercano = np.where(dist_local <= dist_visit, 0, 1)
    return con_balon, vis, control_claro, equipo_cercano


# ---------------------------------------------------------------- 3. orientación
def sentido_ataque(jug, equipo_slot):
    """+1 si el equipo ataca hacia +x, -1 si hacia -x, por tiempo (portero = más atrasado)."""
    mitades = {"1er tiempo": slice(11185, 90652), "2do tiempo": slice(110808, 190198)}
    sentido = {}
    for nombre, sl in mitades.items():
        for eq in (0, 1):
            x = jug[sl, :, 0][::25][:, equipo_slot == eq]
            x = x[~np.isnan(x).all(axis=1)]
            x_min, x_max = np.nanmin(x, axis=1).mean(), np.nanmax(x, axis=1).mean()
            sentido[(nombre, eq)] = 1 if x_min < (LARGO - x_max) else -1
    segunda = np.arange(N_FRAMES) >= INICIO_2T
    s_loc = np.where(segunda, sentido[("2do tiempo", 0)], sentido[("1er tiempo", 0)])
    s_vis = np.where(segunda, sentido[("2do tiempo", 1)], sentido[("1er tiempo", 1)])
    return sentido, s_loc, s_vis


# ---------------------------------------------------------------- 4. features
def forma_equipo(jug, equipo_slot, eq, signo):
    x = jug[:, :, 0][:, equipo_slot == eq]
    y = jug[:, :, 1][:, equipo_slot == eq]
    v = jug[:, :, 2][:, equipo_slot == eq]
    x = np.where(signo[:, None] == 1, x, LARGO - x)      # x grande = cerca de la portería rival
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames sin jugadores -> NaN
        return {"cx": np.nanmean(x, axis=1),
                "sx": np.nanstd(x, axis=1),
                "sy": np.nanstd(y, axis=1),
                "linea_def": np.nanpercentile(x, 20, axis=1),
                "vel": np.nanmean(v, axis=1)}


def cambio_1s(a):
    out = np.full_like(a, np.nan)
    out[25:] = a[25:] - a[:-25]
    return out


def features(jug, equipo_slot, s_loc, s_vis):
    cols = {}
    for pref, f in (("L", forma_equipo(jug, equipo_slot, 0, s_loc)),
                    ("V", forma_equipo(jug, equipo_slot, 1, s_vis))):
        for nombre, a in f.items():
            cols[f"{pref}_{nombre}"] = a
        cols[f"{pref}_dcx_1s"] = cambio_1s(f["cx"])
    feats = pd.DataFrame(cols)
    partes = [feats]
    for nombre, n in {"5s": 125, "15s": 375, "30s": 751}.items():
        media = feats[BASE].rolling(n, center=True, min_periods=1).mean()
        partes.append(media.add_suffix(f"_m{nombre}"))
    return feats, pd.concat(partes, axis=1)


# ---------------------------------------------------------------- 5. modelo de forma + calibración
def prob_local(feats_v, calificado, con_balon, control_claro, equipo_cercano):
    """P(local) por frame (fuera de muestra en las semillas) y tramo de cada semilla."""
    idx = np.flatnonzero(control_claro)
    y = equipo_cercano[idx]
    X_v = feats_v.iloc[idx]

    # 6 bloques contiguos con igual número de frames calificados
    orden = np.flatnonzero(calificado)
    bloque_de_frame = np.full(N_FRAMES, -1)
    bloque_de_frame[orden] = np.minimum(np.arange(len(orden)) * 6 // len(orden), 5)
    bloque = bloque_de_frame[idx]

    # Peso: cada tramo de distancia al hueco suma lo mismo
    dist_hueco = distance_transform_edt(con_balon | ~calificado) / 25
    tramos = pd.cut(dist_hueco[idx], [0, 2, 5, 10, np.inf], labels=TRAMOS, include_lowest=True)
    t = pd.Series(np.asarray(tramos))
    peso = (1 / t.map(t.value_counts()).astype(float)).values
    peso = peso / peso.mean()

    modelo_final = modelo_xgb().fit(X_v, y, sample_weight=peso)
    p_local = 1 - modelo_final.predict_proba(feats_v)[:, 1]      # y=1 es visitante

    prob_oof = np.zeros(len(y))
    for b in range(6):
        train, test = bloque != b, bloque == b
        m = modelo_xgb().fit(X_v[train], y[train], sample_weight=peso[train])
        prob_oof[test] = m.predict_proba(X_v[test])[:, 1]
    p_local[idx] = 1 - prob_oof
    return p_local, idx, y, t.values


def calibrar(p_local, idx, y, tramo):
    """Temperature scaling: T que minimiza el log-loss en semillas a < 5 s de un hueco."""
    p_semilla = p_local[idx]
    real_local = (y == 0)
    cerca_hueco = np.isin(tramo, ["0-2 s", "2-5 s"])
    z = logit(np.clip(p_semilla[cerca_hueco], EPS, 1 - EPS))

    def perdida(T):
        return log_loss(real_local[cerca_hueco], expit(z / T), labels=[False, True])

    T = minimize_scalar(perdida, bounds=(0.5, 10), method="bounded").x
    return expit(logit(np.clip(p_local, EPS, 1 - EPS)) / T), T


# ---------------------------------------------------------------- 6. pausas (D)
def pausas(bal, con_balon, calificado, feats):
    tags = cargar_tags()

    def es_balon_parado(f):
        return (f["code"] or "").strip().lower() == "set pieces"

    inicio_bp = np.sort(np.array([f["start_frame"] for f in tags if es_balon_parado(f)]))
    inicio_ko = np.sort(np.array([f["start_frame"] for f in tags if es_balon_parado(f)
                                  and (f["tags"].get("Type") or "").strip().lower() == "kick off (start)"]))

    # Balón visible a más de 1 m fuera de la cancha
    dist_fuera = np.maximum.reduce([-bal[:, 0], bal[:, 0] - LARGO, -bal[:, 1], bal[:, 1] - ANCHO])
    balon_fuera = con_balon & (dist_fuera > 1)

    # Desde el inicio calificado hasta el primer kick off (start) de cada tiempo
    antes_ko = np.zeros(N_FRAMES, bool)
    for ini, fin in TIEMPOS:
        ko_tiempo = inicio_ko[(inicio_ko >= ini) & (inicio_ko < fin)]
        if len(ko_tiempo):
            antes_ko[ini:ko_tiempo[0]] = True

    lista = []
    for ini, fin in TIEMPOS:
        comienzos, largos = huecos(~con_balon, ini, fin)
        lista += [(a, a + n) for a, n in zip(comienzos, largos)]

    def dist_borde(i):
        if i < 0 or i >= N_FRAMES or not con_balon[i]:
            return np.nan
        x, y = bal[i, 0], bal[i, 1]
        return min(x, LARGO - x, y, ANCHO - y)

    def dist_centro(i):
        if i < 0 or i >= N_FRAMES or not con_balon[i]:
            return np.nan
        return np.hypot(bal[i, 0] - LARGO / 2, bal[i, 1] - ANCHO / 2)

    vel = feats[["L_vel", "V_vel"]].mean(axis=1).values      # solo para el chequeo
    g = pd.DataFrame([{
        "dur_s": (b - a) / 25,
        "d_salida": dist_borde(a - 1),
        "d_entrada": dist_borde(b),
        "entra_centro": dist_centro(b) < 3,
        "reinicio": np.any((inicio_bp >= a) & (inicio_bp < b + 125)),
        "vel": np.nanmean(vel[a:b]),
    } for a, b in lista])

    es_pausa = ((g.d_salida < 3) | (g.d_entrada < 3) | g.entra_centro | g.reinicio).values
    muerto = balon_fuera | antes_ko
    for (a, b), p in zip(lista, es_pausa):
        if p:
            muerto[a:b] = True
    return muerto & calificado, es_pausa, g


# ---------------------------------------------------------------- 6b. detector de pausas (entrenado con DFL)
def features_d(px, py, equipo):
    """Features sin balón, una fila por frame. La misma fórmula que se usó con el DFL."""
    desp = np.full(px.shape, np.nan, np.float32)
    desp[12:-13] = np.hypot(px[25:] - px[:-25], py[25:] - py[:-25])      # desplazamiento en 1 s, centrado

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.array([np.nanmedian(desp[:, equipo == e], axis=1) for e in (0, 1)])
        sy = np.array([np.nanstd(py[:, equipo == e], axis=1) for e in (0, 1)])
        base = pd.DataFrame({"vel": np.nanmean(v, axis=0), "vel_min": np.nanmin(v, axis=0),
                             "sy": np.nanmean(sy, axis=0)})

    def ventana(s, n):
        return s.rolling(n, center=True, min_periods=1).mean()

    return pd.DataFrame({"vel_3s": ventana(base.vel, 75), "vel_10s": ventana(base.vel, 250),
                         "vel_min_3s": ventana(base.vel_min, 75), "sy_3s": ventana(base.sy, 75)})


def limpiar(m, cierre_s, min_s):
    """Rellena huecos de menos de cierre_s y descarta tramos de menos de min_s."""
    m = m.copy()
    if cierre_s:
        k = int(cierre_s * 25)
        m = binary_closing(np.pad(m, k), structure=np.ones(k, bool))[k:-k]
    inicios, largos = huecos(m, 0, len(m))
    for s, l in zip(inicios, largos):
        if l < min_s * 25:
            m[s:s + l] = False
    return m


def pausas_detector(jug, equipo_slot, muerto, calificado):
    """Regla de huecos + pausas nuevas del detector. La regla conserva siempre la prioridad."""
    z = np.load(DFL_TRAIN)
    detector = modelo_xgb().fit(z["X"], z["y"])
    fd = features_d(jug[:, :, 0], jug[:, :, 1], equipo_slot)
    ok = fd[COLS_D].notna().all(axis=1).values
    p = np.zeros(N_FRAMES)
    p[ok] = detector.predict_proba(fd.loc[ok, COLS_D].to_numpy())[:, 1]
    nuevo = limpiar((p > UMBRAL_D) & calificado, CIERRE_S, MIN_PAUSA_S) & ~muerto
    return muerto | nuevo


# ---------------------------------------------------------------- 7. Viterbi H/A
def viterbi(log_e, trans):
    T, K = log_e.shape
    score = log_e[0].copy()
    atras = np.zeros((T, K), np.int8)
    for t in range(1, T):
        cand = score[:, None] + trans
        atras[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_e[t]
    camino = np.empty(T, np.int8)
    camino[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        camino[t - 1] = atras[t, camino[t]]
    return camino


def segmentar(p_local_cal, vis, control_claro, equipo_cercano, muerto, lam):
    # Evidencia de equipo: semillas casi seguras; visible dudoso = forma + pista débil del más cercano
    p_h = p_local_cal.copy()
    p_h[control_claro] = np.where(equipo_cercano[control_claro] == 0, 0.9, 0.1)
    dudoso = vis & ~control_claro
    q = np.where(equipo_cercano == 0, 0.6, 0.4)
    combinado = p_local_cal * q / (p_local_cal * q + (1 - p_local_cal) * (1 - q))
    p_h[dudoso] = combinado[dudoso]

    log_e2 = np.log(np.clip(np.column_stack([p_h, 1 - p_h]), EPS, 1)) / 25   # 1 s = 1 observación
    trans = np.array([[0, -lam], [-lam, 0]], float)
    etq = np.full(N_FRAMES, D, np.int8)                 # pausas y no calificados: D (restricción dura)
    for ini, fin in TIEMPOS:
        comienzos, largos = huecos(~muerto, ini, fin)   # tramos de juego vivo
        for a, n in zip(comienzos, largos):
            etq[a:a + n] = viterbi(log_e2[a:a + n], trans)
    return etq


def posesiones(etq):
    largos = []
    for ini, fin in TIEMPOS:
        e = etq[ini:fin]
        inicios = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1]
        l = np.diff(np.r_[inicios, len(e)])
        largos.append(l[e[inicios] != D])
    l = np.concatenate(largos)
    return len(l), l.mean() / 25


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "run_v2.csv"))
    ap.add_argument("--sin-detector", action="store_true", help="solo la regla de huecos (regenera la v2)")
    ap.add_argument("--sin-cache", action="store_true", help="leer el ATD crudo e ignorar el caché .npz")
    args = ap.parse_args()

    import scipy, sklearn, xgboost
    print(f"numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__} | "
          f"sklearn {sklearn.__version__} | xgboost {xgboost.__version__}")

    rutas = asset_paths()
    jug, bal = cargar_atd(rutas, usar_cache=not args.sin_cache)
    equipo_slot = cargar_equipos(rutas)
    calificado = np.zeros(N_FRAMES, dtype=bool)
    for ini, fin in TIEMPOS:
        calificado[ini:fin] = True

    con_balon, vis, control_claro, equipo_cercano = semillas(jug, bal, calificado, equipo_slot)
    print(f"Semillas: {control_claro.sum()} ({control_claro[calificado].mean():.1%} de los calificados)")

    sentido, s_loc, s_vis = sentido_ataque(jug, equipo_slot)
    print("Sentido de ataque:", sentido)

    feats, feats_v = features(jug, equipo_slot, s_loc, s_vis)
    p_local, idx, y, tramo = prob_local(feats_v, calificado, con_balon, control_claro, equipo_cercano)
    p_local_cal, T = calibrar(p_local, idx, y, tramo)
    print(f"Temperatura T = {T:.4f}")

    muerto, es_pausa, g = pausas(bal, con_balon, calificado, feats)
    print(f"Huecos pausa: {es_pausa.sum()}/{len(es_pausa)} | D en calificados: {muerto[calificado].mean():.1%}")
    print(f"Velocidad media (m/s): pausa {g.vel[es_pausa].mean():.2f} | vivo {g.vel[~es_pausa].mean():.2f}")

    if not args.sin_detector:
        muerto_d = pausas_detector(jug, equipo_slot, muerto, calificado)
        print(f"D en calificados: regla {muerto[calificado].mean():.1%} | regla + detector {muerto_d[calificado].mean():.1%}")
        muerto = muerto_d

    etq = segmentar(p_local_cal, vis, control_claro, equipo_cercano, muerto, LAMBDA)
    n, media = posesiones(etq)
    print(f"Posesiones: {n} | duración media {media:.1f} s")

    etiquetas = np.array(["H", "A", "D"])[etq]
    print("Reparto (todos los frames):", {k: f"{(etiquetas == k).mean():.1%}" for k in "HAD"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": etiquetas}).to_csv(out, index=False)
    print("Guardado en:", out)


if __name__ == "__main__":
    main()
