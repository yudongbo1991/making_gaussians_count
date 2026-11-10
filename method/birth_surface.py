"""Birth surface for the final analytic method."""

import hashlib
from pathlib import Path
from method.teacher_ellipse import EllipticalTeacherSurface


class Guidance:

    def __init__(self, teacher_ply):
        self.path = Path(teacher_ply)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.surface = EllipticalTeacherSurface(self.path)
