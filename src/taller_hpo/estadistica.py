"""Estadística para comparar configuraciones: intervalos, veredictos y tamaño de muestra.

Un solo número ("A dio 0.83 y B dio 0.81") no dice si la diferencia es real o es la
varianza de la semilla. Este módulo responde con intervalos de confianza t, porque con
pocas semillas la normal subestima la incertidumbre, y con un veredicto en vez de un
p-valor suelto:

- "mejor" o "peor" cuando el intervalo de confianza no cruza el cero.
- "equivalente" cuando el intervalo TOST (las dos pruebas unilaterales de equivalencia,
  de nivel 1 - 2*alfa) cae dentro de la zona de indiferencia (-delta, +delta). Un
  p-valor grande nunca prueba equivalencia; TOST sí, con su propia hipótesis nula.
- "no concluyente" cuando ninguna de las dos se sostiene con las semillas que hay.

Wilcoxon se reporta como referencia, pero el veredicto nunca sale de ahí: con 3 a 10
semillas su p-valor mínimo es demasiado alto (0.0625 con n=5).

`semillas_necesarias` dice cuántas semillas hacen falta para detectar una diferencia, y
`maximo_esperado_normal` cuánto infla el máximo de un buscador el valor de una corrida.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.integrate import quad

# Dominio de alfa de semillas_necesarias: fuera de él, scipy.stats.nct con pocos grados
# de libertad no converge (ver la docstring de semillas_necesarias).
ALFA_MINIMO = 1e-3
ALFA_MAXIMO = 0.5


def _validar_confianza(confianza: float) -> None:
    if not (0.0 < confianza < 1.0):
        raise ValueError(f"confianza debe estar en (0, 1), no {confianza!r}")


@dataclass(frozen=True)
class Resumen:
    n: int
    media: float
    de: float
    ee: float
    ic_bajo: float
    ic_alto: float
    confianza: float


def resumen(valores: Sequence[float], *, confianza: float = 0.95) -> Resumen:
    """Media, desviación estándar y error estándar de `valores`, con intervalo
    de confianza t de Student.

    El intervalo de confianza usa la t (no la normal) porque un taller compara
    con 3 a 10 semillas: con esos grados de libertad la normal da un intervalo
    demasiado angosto.
    """
    _validar_confianza(confianza)
    v = np.asarray(valores, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"resumen necesita una secuencia 1-D, no de forma {v.shape}")
    n = v.size
    if n < 2:
        raise ValueError(f"resumen necesita al menos 2 valores, no {n}")
    no_finitos = np.flatnonzero(~np.isfinite(v))
    if no_finitos.size:
        raise ValueError(f"valores no finitos en los índices {no_finitos.tolist()}")

    media = float(v.mean())
    de = float(v.std(ddof=1))
    ee = de / math.sqrt(n)
    alfa = 1.0 - confianza
    t_critico = float(stats.t.ppf(1.0 - alfa / 2.0, df=n - 1))
    return Resumen(
        n=n,
        media=media,
        de=de,
        ee=ee,
        ic_bajo=media - t_critico * ee,
        ic_alto=media + t_critico * ee,
        confianza=confianza,
    )


@dataclass(frozen=True)
class Pareado:
    n: int
    diferencia_media: float
    de_dif: float
    ic_bajo: float
    ic_alto: float
    p_valor: float
    wilcoxon_p: float | None
    rho: float
    veredicto: str
    delta: float
    direccion: str
    confianza: float


def diferencia_pareada(
    a: Mapping[int, float],
    b: Mapping[int, float],
    *,
    delta: float,
    direccion: str = "maximizar",
    confianza: float = 0.95,
) -> Pareado:
    """Compara `a` contra `b`, emparejados por clave (la semilla que comparten).

    Emparejar por clave (en vez de por posición) evita comparar la semilla 3 de
    `a` con la semilla 7 de `b` si alguna corrida se reordenó o se cayó.
    Claves distintas es un error: no hay con qué emparejar lo que sobra o falta.

    La diferencia es `a - b` ("minimizar" invierte el signo), así que "mejor"
    siempre significa que `a` gana, sin importar si la métrica se maximiza o
    minimiza.

    `confianza` debe ser > 0.5 (no solo estar en (0, 1) como en `resumen`):
    el TOST usa el nivel 1 - 2*alfa, que solo es positivo cuando alfa < 0.5.
    """
    _validar_confianza(confianza)
    if confianza <= 0.5:
        # El TOST necesita el nivel 1 - 2*alfa > 0: con confianza <= 0.5 el intervalo
        # TOST colapsa a un punto o queda invertido, y `_validar_confianza` sola acepta
        # cualquier valor en (0, 1).
        raise ValueError(
            f"confianza debe ser > 0.5 en diferencia_pareada (el TOST necesita "
            f"1 - 2*alfa > 0, con alfa = 1 - confianza), no {confianza!r}"
        )
    if direccion not in ("maximizar", "minimizar"):
        raise ValueError(f"direccion debe ser 'maximizar' o 'minimizar', no {direccion!r}")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta debe ser un número finito > 0, no {delta!r}")

    solo_a = sorted(set(a) - set(b))
    solo_b = sorted(set(b) - set(a))
    if solo_a or solo_b:
        raise ValueError(f"claves distintas entre a y b: solo en a={solo_a}, solo en b={solo_b}")

    claves = sorted(a)
    n = len(claves)
    if n < 2:
        raise ValueError(f"diferencia_pareada necesita al menos 2 claves, no {n}")
    a_vals = np.array([a[k] for k in claves], dtype=float)
    b_vals = np.array([b[k] for k in claves], dtype=float)

    # Un NaN o un inf no debe colarse como significancia falsa.
    malas = [
        k
        for k, x, y in zip(claves, a_vals, b_vals, strict=True)
        if not (math.isfinite(x) and math.isfinite(y))
    ]
    if malas:
        raise ValueError(f"valores no finitos en a o b para las claves {malas}")

    signo = 1.0 if direccion == "maximizar" else -1.0
    diferencias = signo * (a_vals - b_vals)

    media = float(diferencias.mean())
    de_dif = float(diferencias.std(ddof=1))
    ee = de_dif / math.sqrt(n)
    df = n - 1
    alfa = 1.0 - confianza

    if ee > 0.0:
        t_critico = float(stats.t.ppf(1.0 - alfa / 2.0, df=df))
        ic_bajo = media - t_critico * ee
        ic_alto = media + t_critico * ee
        t_stat = media / ee
        p_valor = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    else:
        # Todas las diferencias son idénticas: el intervalo colapsa a un punto y el
        # p-valor es el caso límite (0 si ese punto no es cero, 1 si lo es).
        ic_bajo = media
        ic_alto = media
        p_valor = 0.0 if media != 0.0 else 1.0

    no_nulas = diferencias[diferencias != 0.0]
    if no_nulas.size >= 6:
        wilcoxon_p = float(stats.wilcoxon(diferencias).pvalue)
    else:
        wilcoxon_p = None

    # rho es la correlación que gana el emparejamiento. Con una de las dos constante no
    # está definida y queda como NaN; comparar contra el primer valor detecta los
    # arreglos constantes que `std() == 0.0` no ve en punto flotante.
    if np.all(a_vals == a_vals[0]) or np.all(b_vals == b_vals[0]):
        rho = float("nan")
    else:
        rho = float(stats.pearsonr(a_vals, b_vals).statistic)

    if ic_bajo > 0.0:
        veredicto = "mejor"
    elif ic_alto < 0.0:
        veredicto = "peor"
    else:
        if ee > 0.0:
            # TOST: dos pruebas unilaterales a nivel alfa equivalen a mirar si
            # el intervalo de confianza de nivel (1 - 2*alfa) cae dentro de
            # (-delta, +delta).
            t_critico_tost = float(stats.t.ppf(1.0 - alfa, df=df))
            ic_bajo_tost = media - t_critico_tost * ee
            ic_alto_tost = media + t_critico_tost * ee
        else:
            ic_bajo_tost = media
            ic_alto_tost = media
        if -delta < ic_bajo_tost and ic_alto_tost < delta:
            veredicto = "equivalente"
        else:
            veredicto = "no concluyente"

    return Pareado(
        n=n,
        diferencia_media=media,
        de_dif=de_dif,
        ic_bajo=ic_bajo,
        ic_alto=ic_alto,
        p_valor=p_valor,
        wilcoxon_p=wilcoxon_p,
        rho=rho,
        veredicto=veredicto,
        delta=delta,
        direccion=direccion,
        confianza=confianza,
    )


def semillas_necesarias(
    de_dif: float, delta: float, *, alfa: float = 0.05, potencia: float = 0.8
) -> int:
    """n mínimo (>= 2) para que una t pareada bilateral detecte `delta` con la potencia pedida.

    Usa la t no central (`scipy.stats.nct`) y no la aproximación normal: con pocas
    semillas la t exige más muestras, porque también hay que estimar la varianza. La
    búsqueda arranca en la aproximación normal, sube hasta alcanzar la potencia y baja
    mientras la potencia siga alcanzando.

    Dominio: `alfa` en [1e-3, 0.5]. Fuera de ese rango `scipy.stats.nct` deja de
    converger y daría un número incorrecto en silencio. Si scipy avisa que no convergió,
    o no puede evaluar la potencia como un número finito, lanza `FloatingPointError`.
    Cuando la potencia en n=2 ya es 1 a precisión de máquina, devuelve 2 sin llamar a
    scipy.
    """
    if not math.isfinite(de_dif) or de_dif <= 0:
        raise ValueError(f"de_dif debe ser un número finito > 0, no {de_dif!r}")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta debe ser un número finito > 0, no {delta!r}")
    if not (ALFA_MINIMO <= alfa <= ALFA_MAXIMO):
        raise ValueError(
            f"alfa debe estar en [{ALFA_MINIMO}, {ALFA_MAXIMO}], no {alfa!r}: fuera de ese "
            "rango scipy.stats.nct no converge con pocas semillas y la respuesta no es confiable"
        )
    if not (0.0 < potencia < 1.0):
        raise ValueError(f"potencia debe estar en (0, 1), no {potencia!r}")

    nc_en_dos = delta * math.sqrt(2) / de_dif
    # t crítico de n=2 (1 grado de libertad): con `alfa` diminuto crece mucho, así que
    # nc_en_dos sola no garantiza potencia alta en n=2.
    t_critico_en_dos = float(stats.t.ppf(1.0 - alfa / 2.0, df=1))
    if nc_en_dos >= 1e6 and nc_en_dos >= 1e3 * t_critico_en_dos:
        # Con las dos condiciones la potencia en n=2 ya es 1, y en este rango
        # scipy.stats.nct puede devolver NaN, así que no se le pregunta.
        return 2

    def potencia_de(n: int) -> float:
        df = n - 1
        t_critico = float(stats.t.ppf(1.0 - alfa / 2.0, df=df))
        nc = delta * math.sqrt(n) / de_dif
        # La cola inferior bilateral se pide como sf(t_critico, df, -nc), por simetría,
        # porque nct.cdf(-t_critico, df, nc) da NaN con nc grande y df chico. Si scipy
        # avisa que la serie no convergió, el número puede ser incorrecto y se escala a
        # error.
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", RuntimeWarning)
            superior = float(stats.nct.sf(t_critico, df, nc))
            inferior = float(stats.nct.sf(t_critico, df, -nc))
        potencia_real = superior + inferior
        no_convergio = [a for a in avisos if issubclass(a.category, RuntimeWarning)]
        if no_convergio or not math.isfinite(potencia_real):
            detalle = str(no_convergio[0].message) if no_convergio else "potencia no finita"
            raise FloatingPointError(
                f"scipy.stats.nct no pudo evaluar la potencia con n={n}, nc={nc!r}: {detalle}"
            )
        return potencia_real

    razon = de_dif / delta
    piso_normal = math.ceil(
        (stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(potencia)) ** 2 * razon**2
    )
    n = max(2, piso_normal)
    while potencia_de(n) < potencia:
        n += 1
    while n > 2 and potencia_de(n - 1) >= potencia:
        n -= 1
    return n


def ic_proporcion(
    k: int, n: int, *, metodo: str = "wilson", confianza: float = 0.95
) -> tuple[float, float]:
    """Intervalo de confianza para una proporción k/n.

    Wilson por defecto, porque Wald da ancho 0 en k=0 o k=n. Valida `k` y `n` por su
    cuenta, porque `scipy.stats.binomtest` devuelve NaN en silencio con valores
    inválidos.
    """
    _validar_confianza(confianza)
    if (
        isinstance(k, bool)
        or isinstance(n, bool)
        or not isinstance(k, numbers.Integral)
        or not isinstance(n, numbers.Integral)
    ):
        raise ValueError(f"k y n deben ser enteros, no k={k!r}, n={n!r}")
    if n < 1 or not (0 <= k <= n):
        raise ValueError(f"se necesita 0 <= k <= n y n >= 1, no k={k!r}, n={n!r}")
    resultado = stats.binomtest(k, n)
    ic = resultado.proportion_ci(confidence_level=confianza, method=metodo)
    return float(ic.low), float(ic.high)


def maximo_esperado_normal(t: int) -> float:
    """E[max de t normales estándar iid] = t * integral(x * phi(x) * Phi(x)^(t-1)).

    Con t = 1 devuelve 0 sin integrar. Acepta enteros de numpy, pero no `bool`.
    """
    if isinstance(t, bool) or not isinstance(t, numbers.Integral) or t < 1:
        raise ValueError(f"t debe ser un entero >= 1, no {t!r}")
    t = int(t)
    if t == 1:
        return 0.0

    def integrando(x: float) -> float:
        return t * x * stats.norm.pdf(x) * stats.norm.cdf(x) ** (t - 1)

    valor, _ = quad(integrando, -np.inf, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
    return float(valor)


def holm(p_valores: Sequence[float]) -> list[float]:
    """Ajuste de Holm-Bonferroni (step-down), devuelto en el orden de entrada.

    Menos conservador que Bonferroni simple: cada p-valor ordenado se compara
    contra una fracción distinta de alfa, no siempre 1/m. El resultado es
    monótono no decreciente en el orden de los p-valores originales (de otro
    modo, un p más chico podría terminar con un ajustado más grande).
    """
    p = np.asarray(p_valores, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p_valores debe ser una secuencia 1-D, no de forma {p.shape}")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"p_valores deben estar en [0, 1], no {p_valores!r}")
    m = p.size
    orden = np.argsort(p)

    ajustados_ordenados = np.empty(m, dtype=float)
    maximo_hasta_ahora = 0.0
    for rango, indice in enumerate(orden):
        candidato = (m - rango) * p[indice]
        maximo_hasta_ahora = max(maximo_hasta_ahora, candidato)
        ajustados_ordenados[rango] = min(maximo_hasta_ahora, 1.0)

    resultado = np.empty(m, dtype=float)
    resultado[orden] = ajustados_ordenados
    return resultado.tolist()
