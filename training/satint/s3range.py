"""
A read-only file over HTTP range requests for h5py: small reads (HDF5 metadata) fetch a whole
block, large reads (a compressed chunk) fetch exactly what is asked for in one request.
"""
import bisect
import io
import time

import requests

_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
        s.mount("https://", a)
        _SESSION = s
    return _SESSION


class RangeFile(io.RawIOBase):
    def __init__(self, url, size=None, block=64 << 10, s=None):
        self.url, self.block = url, block
        self.s = s or session()
        if size is None:
            h = self._retry(lambda: self.s.head(url, timeout=60))
            size = int(h.headers["Content-Length"])
        self.size = int(size)
        self.pos = 0
        self.starts, self.bufs = [], []          # sorted, non-overlapping cached segments
        self.fetched = 0
        self.requests = 0

    def _retry(self, fn):
        last = None
        for attempt in range(6):
            try:
                r = fn()
                if r.status_code in (200, 206):
                    return r
                last = IOError(f"HTTP {r.status_code} {self.url}")
                if r.status_code in (403, 404):
                    break
            except requests.RequestException as exc:
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise IOError(f"{self.url}: {last}")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _cached(self, a, b):
        """Bytes a..b-1 if one cached segment holds them."""
        i = bisect.bisect_right(self.starts, a) - 1
        if i >= 0:
            s0, buf = self.starts[i], self.bufs[i]
            if a >= s0 and b <= s0 + len(buf):
                return buf[a - s0:b - s0]
        return None

    def _fetch(self, a, b):
        r = self._retry(lambda: self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120))
        data = r.content
        if r.status_code == 200:                       # the server ignored the range
            data = data[a:b]
        self.fetched += len(data)
        self.requests += 1
        i = bisect.bisect_left(self.starts, a)
        self.starts.insert(i, a)
        self.bufs.insert(i, data)
        return data

    def readinto(self, buf):
        n = len(buf)
        if self.pos >= self.size or n == 0:
            return 0
        n = min(n, self.size - self.pos)
        a, b = self.pos, self.pos + n
        data = self._cached(a, b)
        if data is None:
            if n >= self.block:
                data = self._fetch(a, b)
            else:
                a0 = (a // self.block) * self.block
                b0 = min(self.size, max(b, a0 + self.block))
                if b0 < b:
                    b0 = b
                seg = self._fetch(a0, b0)
                data = seg[a - a0:b - a0]
        buf[:n] = data
        self.pos += n
        return n
