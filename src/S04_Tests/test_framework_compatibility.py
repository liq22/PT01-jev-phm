"""Direct regression checks for legacy/current Skills and local wrapper preservation."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / ".agent/scripts"


class FrameworkCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.skill = self.root / ".agent/skills/00-router"
        self.skill.mkdir(parents=True)
        self.write_skill(["## Purpose", "## Workflow", "## Output Contract", "## Boundaries"])

    def write_skill(self, sections):
        text = "---\nname: 00-router\ndescription: Route the requested research work using its existing local project authority and the actual evidence available.\n---\n\n"
        text += "\n\n".join(section + "\n\nRead paper/paper.yaml when applicable." for section in sections)
        (self.skill / "SKILL.md").write_text(text, encoding="utf-8")


    def run_generator(self, *args):
        return subprocess.run(["python", str(SCRIPTS / "generate_agent_skill_wrappers.py"), "--repo-root", str(self.root), *args], text=True, capture_output=True)

    def sentinel(self, host="codex", name="local-analysis"):
        path = self.root / f".{host}/skills/{name}/SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Local author-maintained Skill.\n")
        return path








    def test_default_shell_preserves_local_skill(self):
        local = self.sentinel()
        env = dict(os.environ)
        env.pop("CLEAN_ENV", None)
        result = subprocess.run(["bash", str(SCRIPTS / "sync_agent_skill_wrappers.sh"), str(self.root), "both"], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(local.read_text(), "Local author-maintained Skill.\n")

    def test_explicit_clean_preserves_local_skill_and_adjacent_asset(self):
        self.assertEqual(self.run_generator().returncode, 0)
        local = self.sentinel()
        asset = self.root / ".codex/skills/00-router/notes.md"
        asset.write_text("Local extension.\n")
        # Without its canonical source, an orphan wrapper is not safe to delete.
        stale = self.root / ".codex/skills/stale/SKILL.md"
        stale.parent.mkdir()
        stale.write_text((asset.parent / "SKILL.md").read_text())
        result = self.run_generator("--clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(stale.exists())
        self.assertEqual(local.read_text(), "Local author-maintained Skill.\n")
        self.assertEqual(asset.read_text(), "Local extension.\n")
        self.assertTrue((asset.parent / "SKILL.md").exists())

    def test_local_same_name_conflict_fails_and_preserves_content(self):
        local = self.sentinel(name="00-router")
        for flags in [(), ("--clean",)]:
            result = self.run_generator(*flags)
            self.assertEqual(result.returncode, 2)
            self.assertIn("preserving local Skill", result.stdout)
            self.assertEqual(local.read_text(), "Local author-maintained Skill.\n")

    def test_symlinked_skill_is_not_overwritten(self):
        outside = self.root / "local-target"
        outside.mkdir()
        target = outside / "SKILL.md"
        target.write_text("Do not replace.\n")
        host = self.root / ".codex/skills"
        host.mkdir(parents=True)
        (host / "00-router").symlink_to(outside, target_is_directory=True)
        result = self.run_generator("--clean")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(target.read_text(), "Do not replace.\n")

    def test_symlinked_host_ancestor_is_not_modified(self):
        outside = self.root / "outside-host"
        target = outside / "skills/00-router/SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_text("Author-owned host data.\n")
        (self.root / ".codex").symlink_to(outside, target_is_directory=True)
        for flags in [(), ("--clean",)]:
            result = self.run_generator(*flags)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), "Author-owned host data.\n")

    def test_edited_generated_metadata_is_preserved_and_reported(self):
        self.assertEqual(self.run_generator().returncode, 0)
        path = self.root / ".codex/skills/00-router/SKILL.md"
        edited = path.read_text().replace("name: 00-router", "name: 00-router\nallowed-tools: Read")
        path.write_text(edited)
        for flags in [(), ("--clean",)]:
            result = self.run_generator(*flags)
            self.assertEqual(result.returncode, 2)
            self.assertIn("reconcile it manually", result.stdout)
            self.assertEqual(path.read_text(), edited)

    def test_host_override_and_wrapper_validation_remain_compatible(self):
        overrides = self.skill / "wrappers"
        overrides.mkdir()
        (overrides / "claude.yaml").write_text("disable-model-invocation: true\n")
        self.assertEqual(self.run_generator().returncode, 0)
        result = subprocess.run(["python", str(SCRIPTS / "validate_agent_skill_wrappers.py"), "--canonical", str(self.root / ".agent/skills"), "--claude", str(self.root / ".claude/skills"), "--codex", str(self.root / ".codex/skills")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


    def test_known_legacy_generated_marker_is_recognized(self):
        self.assertEqual(self.run_generator().returncode, 0)
        path = self.root / ".codex/skills/00-router/SKILL.md"
        original = path.read_text()
        legacy = original.replace(
            "Generated by .agent/scripts/sync_agent_skill_wrappers.sh. Do not edit manually.",
            "Generated wrapper. Canonical content lives under .agent/skills/.",
        )
        self.assertNotEqual(original, legacy)
        path.write_text(legacy)
        result = self.run_generator("--clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(path.read_text(), original)
        edited = legacy.replace("name: 00-router", "name: 00-router\nallowed-tools: Read")
        path.write_text(edited)
        result = self.run_generator("--clean")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_text(), edited)

    def test_strict_host_exposure_still_rejects_local_extras(self):
        self.assertEqual(self.run_generator().returncode, 0)
        local = self.sentinel()
        result = subprocess.run(["python", str(SCRIPTS / "validate_agent_skill_wrappers.py"), "--canonical", str(self.root / ".agent/skills"), "--claude", str(self.root / ".claude/skills"), "--codex", str(self.root / ".codex/skills")], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(local.read_text(), "Local author-maintained Skill.\n")


if __name__ == "__main__":
    unittest.main()
