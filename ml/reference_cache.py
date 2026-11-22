"""Atomic local cache for the embeddings of catalog reference photographs."""

import hashlib
import os
from pathlib import Path
import pickle
import tempfile

import torch


class ReferenceCache:
    def __init__(self, directory, implementation, weights):
        self.directory = Path(directory) if directory else None
        self.version = None
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            signature = hashlib.sha256(str(torch.__version__).encode())
            for file in implementation:
                signature.update(Path(file).read_bytes())
            for file in weights:
                path = Path(file)
                signature.update(str(path).encode())
                if path.is_file():
                    stat = path.stat()
                    signature.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
            self.version = signature.hexdigest()
        except OSError:
            # The catalog still works when the derivative cache cannot be written.
            self.directory = None

    def _record(self, photograph):
        path = Path(photograph).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_mtime_ns, stat.st_size, self.version)
        return self.directory / f"{path.name}.pt", identity

    def vectors_for(self, photograph, compute_global, compute_parts=None):
        if self.directory is None:
            return compute_global(), (compute_parts() if compute_parts else None) or {}
        try:
            file, identity = self._record(photograph)
        except OSError:
            return compute_global(), (compute_parts() if compute_parts else None) or {}
        vectors = parts = None
        try:
            record = torch.load(file, map_location="cpu", weights_only=True)
            if (isinstance(record, dict) and record.get("key") == identity and
                    isinstance(record.get("global"), torch.Tensor)):
                vectors = record["global"]
                if isinstance(record.get("parts"), dict):
                    parts = record["parts"]
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
            pass
        needs_global = vectors is None
        needs_parts = compute_parts is not None and parts is None
        if not needs_global and not needs_parts:
            return vectors, parts or {}
        if needs_global:
            vectors = compute_global()
        if needs_parts:
            parts = compute_parts()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                torch.save({"key": identity, "global": vectors, "parts": parts}, output)
            os.replace(temporary, file)
        except (OSError, RuntimeError):
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return vectors, parts or {}
