"""Random number generation compatible with R's default generator.

The R implementations of scTenifoldNet and scTenifoldKnk draw the cells of
each network with ``sample()`` and the initial CP factors with ``rnorm()``
after ``set.seed(seed)``. :class:`RRandom` reproduces those streams exactly
(R's defaults: Mersenne-Twister, Inversion and Rejection sampling), so both
implementations give the same results for the same seed.
"""
from typing import Optional

import numpy as np

__all__ = ["RRandom"]

_N_MT = 624
_BIG = 134217728  # 2 ** 27
_I2_32M1 = 2.328306437080797e-10  # 1 / (2 ** 32 - 1)


def _mt_key(seed: int) -> np.ndarray:
    # set.seed(): initial scrambling of the seed with an LCG, then one LCG
    # step per element of .Random.seed. Its first element is the position in
    # the state (624, so that the first draw regenerates it).
    seed = seed & 0xFFFFFFFF
    for _ in range(50):
        seed = (69069 * seed + 1) & 0xFFFFFFFF
    key = np.empty(_N_MT + 1, dtype=np.uint32)
    for j in range(_N_MT + 1):
        seed = (69069 * seed + 1) & 0xFFFFFFFF
        key[j] = seed
    return key[1:]


def _qnorm(p: np.ndarray) -> np.ndarray:
    # Algorithm AS 241 (Wichura, 1988), as in R's qnorm(p) for lower-tail,
    # non-log probabilities
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    out = np.empty_like(p)

    central = np.abs(q) <= 0.425
    if central.any():
        qc = q[central]
        r = 0.180625 - qc * qc
        out[central] = (
            qc * (((((((r * 2509.0809287301226727 +
                        33430.575583588128105) * r + 67265.770927008700853) * r +
                      45921.953931549871457) * r + 13731.693765509461125) * r +
                    1971.5909503065514427) * r + 133.14166789178437745) * r +
                  3.387132872796366608)
            / (((((((r * 5226.495278852545925 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.)
        )

    tail = ~central
    if tail.any():
        qt = q[tail]
        pt = p[tail]
        r = np.sqrt(-np.log(np.where(qt > 0, 1 - pt, pt)))
        val = np.empty_like(r)
        near = r <= 5.
        rn = r[near] - 1.6
        val[near] = (
            (((((((rn * 7.7454501427834140764e-4 +
                   .0227238449892691845833) * rn + .24178072517745061177) *
                 rn + 1.27045825245236838258) * rn +
                3.64784832476320460504) * rn + 5.7694972214606914055) *
              rn + 4.6303378461565452959) * rn +
             1.42343711074968357734)
            / (((((((rn *
                     1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                    rn + .0151986665636164571966) * rn +
                   .14810397642748007459) * rn + .68976733498510000455) *
                 rn + 1.6763848301838038494) * rn +
                2.05319162663775882187) * rn + 1.)
        )
        rf = r[~near] - 5.
        val[~near] = (
            (((((((rf * 2.01033439929228813265e-7 +
                   2.71155556874348757815e-5) * rf +
                  .0012426609473880784386) * rf + .026532189526576123093) *
                rf + .29656057182850489123) * rf +
               1.7848265399172913358) * rf + 5.4637849111641143699) *
             rf + 6.6579046435011037772)
            / (((((((rf *
                     2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                    rf + 1.8463183175100546818e-5) * rf +
                   7.868691311456132591e-4) * rf + .0148753612908506148525)
                 * rf + .13692988092273580531) * rf +
                .59983220655588793769) * rf + 1.)
        )
        out[tail] = np.where(qt < 0, -val, val)
    return out


class RRandom:
    """R's default random number generator.

    ``RRandom(seed)`` is equivalent to ``set.seed(seed)`` in R; the methods
    return the same values as R's ``runif``, ``rnorm`` and ``sample``.

    Parameters
    ----------
    seed
        Integer seed, as passed to ``set.seed``. If None, a seed is drawn
        from the operating system.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self._mt = np.random.MT19937()
        self._mt.state = {"bit_generator": "MT19937",
                          "state": {"key": _mt_key(int(seed)), "pos": _N_MT}}

    def unif_rand(self, n: int) -> np.ndarray:
        """``n`` uniform draws in (0, 1), as R's ``unif_rand``."""
        u = self._mt.random_raw(n).astype(float) * 2.3283064365386963e-10
        # fixup(): keep draws strictly inside (0, 1)
        u[u <= 0] = 0.5 * _I2_32M1
        u[1 - u <= 0] = 1 - 0.5 * _I2_32M1
        return u

    def rnorm(self, n: int) -> np.ndarray:
        """``n`` standard normal draws, as R's ``rnorm(n)``."""
        # Inversion: two uniforms per draw for extra precision
        u = self.unif_rand(2 * n)
        u = np.floor(_BIG * u[0::2]) + u[1::2]
        return _qnorm(u / _BIG)

    def _unif_index(self, n: int) -> int:
        # R_unif_index(): rejection sampling from the integers below the next
        # power of two, built from 16 bits per uniform draw
        if n <= 1:
            return 0
        bits = int(np.ceil(np.log2(n)))
        chunks = bits // 16 + 1
        mask = (1 << bits) - 1
        while True:
            v = 0
            for u in self.unif_rand(chunks):
                v = 65536 * v + int(np.floor(u * 65536))
            v &= mask
            if v < n:
                return v

    def sample(self, n: int, size: int, replace: bool = True) -> np.ndarray:
        """``size`` draws from ``0, ..., n - 1``.

        Equivalent to ``sample(n, size, replace) - 1`` in R.
        """
        if replace or size < 2:
            return np.array([self._unif_index(n) for _ in range(size)], dtype=int)
        if size > n:
            raise ValueError("cannot take a sample larger than the population when replace is False")
        out = np.empty(size, dtype=int)
        if n > 1e7 and size <= n / 2:
            # sample2(): draw until 'size' distinct values are found
            seen = set()
            i = 0
            while i < size:
                v = self._unif_index(n)
                if v not in seen:
                    seen.add(v)
                    out[i] = v
                    i += 1
            return out
        # Partial shuffle: each drawn value is replaced by the last remaining
        pool = np.arange(n)
        remaining = n
        for i in range(size):
            j = self._unif_index(remaining)
            out[i] = pool[j]
            remaining -= 1
            pool[j] = pool[remaining]
        return out
