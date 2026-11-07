from pathlib import Path
from tempfile import TemporaryDirectory

from editables import EditableProject
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or version != "editable":
            return
        self.temporary = TemporaryDirectory(prefix="yapit-editable-")
        project = EditableProject("yapit", self.root)
        project.map("yapit", Path(self.root) / "src")
        for filename, content in project.files():
            path = Path(self.temporary.name) / filename
            path.write_text(content, encoding="utf-8")
            build_data["force_include"][str(path)] = filename
        build_data["dependencies"].append("editables>=0.3,<1")

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        if self.target_name == "wheel" and version == "editable":
            self.temporary.cleanup()
