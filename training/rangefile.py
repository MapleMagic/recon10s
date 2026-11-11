"""A read-only file object over HTTP range requests, with a block cache (for h5py on huge remote files)."""
import io
import time
from collections import OrderedDict

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url, block=4 << 20, max_blocks=64, session=None):
        self.url, self.block, self.max_blocks = url, block, max_blocks
        self.s = session or requests.Session()
        for attempt in range(8):
            try:
                h = self.s.head(url, timeout=60)
                h.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 7:
                    raise
                time.sleep(2 + 2 * attempt)
                self.s = requests.Session()
        self.size = int(h.headers["Content-Length"])
        self.pos = 0
        self.cache = OrderedDict()
        self.fetched = 0
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _get(self, k):
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        a = k * self.block
        b = min(self.size, a + self.block) - 1
        for attempt in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    break
            except requests.RequestException:
                self.s = requests.Session()
            time.sleep(2 + 3 * attempt)
        else:
            raise IOError(f"range {a}-{b} failed")
        self.fetched += len(r.content)
        self.requests += 1
        self.cache[k] = r.content
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return r.content

    def readinto(self, buf):
        n = len(buf)
        if self.pos >= self.size or n == 0:
            return 0
        n = min(n, self.size - self.pos)
        out = memoryview(buf)
        done = 0
        while done < n:
            k, off = divmod(self.pos + done, self.block)
            data = self._get(k)
            take = min(len(data) - off, n - done)
            out[done:done + take] = data[off:off + take]
            done += take
        self.pos += n
        return n
