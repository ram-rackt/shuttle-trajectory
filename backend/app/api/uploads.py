"""Receives a video upload by streaming the multipart body straight into the
store's tmp/ folder, hashing it on the way.

Starlette's own form parsing (UploadFile) first spools the whole body to the
operating system's temp folder: the size limit could only be checked once
everything had arrived, and every upload was written to disk twice. Here the
limit stops the upload as soon as it is exceeded, and the video is written
once, inside storage/.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request
from python_multipart.multipart import MultipartParser, parse_options_header

MAX_FIELD_BYTES = 64 * 1024
"""Limit for the small text fields sent with the video."""
FORM_OVERHEAD_BYTES = 256 * 1024
"""Room for those fields and the multipart framing, on top of the video size limit."""


@dataclass
class ReceivedUpload:
    fields: dict[str, str]
    """The text fields, by name."""
    filename: str | None
    """The video's file name (None if the form had no video)."""
    size: int
    sha256: str


FileCheck = Callable[[str, dict[str, str]], None]
"""Called with the file name and the fields received so far when the video
part starts, before any of it is stored. Raises HTTPException to refuse it."""


async def receive_upload(
    request: Request, dest: Path, max_bytes: int, file_field: str, check_file: FileCheck
) -> ReceivedUpload:
    """Writes the `file_field` part of a multipart/form-data body to `dest`.

    Raises HTTPException (413) as soon as the video exceeds `max_bytes`. On any
    failure `dest` may be left partly written; the caller removes it.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes + FORM_OVERHEAD_BYTES:
        raise HTTPException(413, "Video is too large.")  # refused before reading anything
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise HTTPException(422, "Send the video as multipart/form-data.")

    reader = _FormReader(dest, max_bytes, file_field, check_file)
    parser = MultipartParser(boundary, reader.callbacks())
    try:
        async for chunk in request.stream():
            parser.write(chunk)
        parser.finalize()
    finally:
        reader.close()
    return ReceivedUpload(reader.fields, reader.filename, reader.size, reader.digest.hexdigest())


class _FormReader:
    """Multipart parser callbacks: text fields into a dict, the video to disk."""

    def __init__(self, dest: Path, max_bytes: int, file_field: str, check_file: FileCheck) -> None:
        self.fields: dict[str, str] = {}
        self.filename: str | None = None
        self.size = 0
        self.digest = hashlib.sha256()
        self._dest = dest
        self._max_bytes = max_bytes
        self._file_field = file_field
        self._check_file = check_file
        self._out = None
        self._headers: dict[bytes, bytes] = {}
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._name: str | None = None
        self._value = bytearray()
        self._in_file = False

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value_data,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def close(self) -> None:
        if self._out is not None and not self._out.closed:
            self._out.close()

    def _part_begin(self) -> None:
        self._headers, self._name, self._value, self._in_file = {}, None, bytearray(), False

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _header_value_data(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _header_end(self) -> None:
        self._headers[bytes(self._header_name).lower()] = bytes(self._header_value)
        self._header_name, self._header_value = bytearray(), bytearray()

    def _headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get(b"content-disposition", b""))
        self._name = params.get(b"name", b"").decode("utf-8", "replace")
        if self._name != self._file_field or b"filename" not in params:
            return
        if self._out is not None:
            raise HTTPException(422, "Send one video at a time.")
        self.filename = params[b"filename"].decode("utf-8", "replace")
        self._check_file(self.filename, self.fields)
        self._out = self._dest.open("wb")
        self._in_file = True

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._in_file:
            self.size += len(chunk)
            if self.size > self._max_bytes:
                raise HTTPException(413, "Video is too large.")
            self.digest.update(chunk)
            self._out.write(chunk)  # type: ignore[union-attr]
        else:
            self._value += chunk
            if len(self._value) > MAX_FIELD_BYTES:
                raise HTTPException(413, f"The form field {self._name!r} is too large.")

    def _part_end(self) -> None:
        if self._in_file:
            self._out.close()  # type: ignore[union-attr]
            self._in_file = False
        elif self._name:
            self.fields[self._name] = self._value.decode("utf-8", "replace")
