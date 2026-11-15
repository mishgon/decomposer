"""Read immutable Wikipedia JSONL pages using a disk-backed URL/offset index."""
import argparse
import json
from pathlib import Path
import sqlite3


def signature(source):
    stat = source.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def build(source):
    index = source.with_suffix(".sqlite")
    temporary = index.with_suffix(".sqlite.building")
    original = signature(source)
    # A failed build never replaces a complete index.
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db, source.open("rb") as stream:
        db.execute("CREATE TABLE pages (url TEXT PRIMARY KEY, offset INTEGER) WITHOUT ROWID")
        db.execute("CREATE TABLE metadata (signature TEXT)")
        count = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            page = json.loads(line)
            db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?)", (page["url"], offset))
            count += 1
            if count % 100_000 == 0:
                db.commit()
                print(f"Indexed {count} pages", flush=True)
        if signature(source) != original:
            raise ValueError("Page file changed while indexing")
        db.execute("INSERT INTO metadata VALUES (?)", (original,))
    temporary.replace(index)
    print(f"Page index ready: {index}", flush=True)


class Pages:
    def __init__(self, source):
        self.source = source
        self.uri = source.with_suffix(".sqlite").resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(self.uri, uri=True) as db:
            if db.execute("SELECT signature FROM metadata").fetchone()[0] != signature(source):
                raise ValueError("Page file changed; rebuild its index")
            self.count = db.execute("SELECT count(*) FROM pages").fetchone()[0]

    def __len__(self):
        return self.count

    def get(self, url):
        # Each request owns its connection and file cursor (FastAPI uses threads).
        with sqlite3.connect(self.uri, uri=True) as db:
            row = db.execute("SELECT offset FROM pages WHERE url=?", (url,)).fetchone()
            if row is None and "_" in url:
                # Wikipedia titles use underscores and encoded spaces interchangeably.
                row = db.execute("SELECT offset FROM pages WHERE url=?", (url.replace("_", "%20"),)).fetchone()
        if row is None:
            return None
        with self.source.open("rb") as stream:
            stream.seek(row[0])
            return json.loads(stream.readline())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    build(parser.parse_args().source)
