"""Verify safe regeneration against repository history using isolated Git fixtures."""
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / ".agent/scripts/generate_agent_skill_wrappers.py"

class RefreshTests(unittest.TestCase):
    def run_case(self, variant, case, clean):
        spec = importlib.util.spec_from_file_location('generator', GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            source = repo / '.agent/skills/00-router/SKILL.md'
            source.parent.mkdir(parents=True)
            source.write_text('---\nname: 00-router\ndescription: Original source\n---\nCanonical body\n')
            def git(*args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            def commit():
                git('add', '.')
                git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')
            def generate(clean_arg=False):
                with contextlib.redirect_stdout(io.StringIO()):
                    args = ['--repo-root', str(repo)] + (['--clean'] if clean_arg else [])
                    if variant == 'simple':
                        with mock.patch.object(sys, 'argv', ['generator', *args]):
                            return module.main()
                    return module.main(args)
            self.assertEqual(generate(), 0)
            wrappers = [repo / f'.{host}/skills/00-router/SKILL.md' for host in ('claude', 'codex')]
            asset = wrappers[0].parent / 'notes.txt'
            asset.write_text('User asset remains')
            if case == 'legacy':
                for wrapper in wrappers:
                    wrapper.write_text(wrapper.read_text().replace(module.PROVENANCE, 'Generated wrapper. Canonical content lives under .agent/skills/.'))
            if case != 'no_git':
                git('init', '-q')
                if case != 'missing_baseline':
                    commit()
            if case == 'missing_historical_source':
                (repo / '.gitignore').write_text('/.agent/skills/00-router/SKILL.md\n')
                git('rm', '--cached', str(source.relative_to(repo)))
                commit()
                for wrapper in wrappers:
                    wrapper.write_text(wrapper.read_text() + '\n')
                commit()
            if case == 'override':
                override = source.parent / 'wrappers/claude.yaml'
                override.parent.mkdir()
                override.write_text('argument-hint: new hint\n')
            elif case == 'existing_override':
                override = source.parent / 'wrappers/claude.yaml'
                override.parent.mkdir()
                override.write_text('argument-hint: old hint\n')
                self.assertEqual(generate(), 0)
                commit()
                override.write_text('argument-hint: new hint\n')
            elif case != 'stable':
                source.write_text(source.read_text().replace('Original source', 'Updated source'))
            if case == 'canonical_committed':
                commit()
            if case in ('dirty_edit', 'committed_edit'):
                for wrapper in wrappers:
                    wrapper.write_text(wrapper.read_text().replace('Original source', 'User description'))
                if case == 'committed_edit':
                    commit()
            if case == 'body_edit':
                for wrapper in wrappers:
                    wrapper.write_text(wrapper.read_text() + '\nUser instructions\n')
            if case == 'wrong_target':
                for wrapper in wrappers:
                    wrapper.write_text(wrapper.read_text().replace('@../../../.agent', '@../../../other'))
            if case in ('missing_object', 'missing_object_plain'):
                oid = git('rev-parse', 'HEAD:.agent/skills/00-router/SKILL.md').decode().strip()
                (repo / '.git/objects' / oid[:2] / oid[2:]).unlink()
                if case == 'missing_object':
                    git('config', 'remote.origin.promisor', 'true')
                    git('config', 'remote.origin.url', 'https://example.invalid/must-not-fetch')
            partial_keys = {
                'partial_filter': ('remote.origin.partialclonefilter', 'blob:none'),
                'partial_extension': ('extensions.partialclone', 'origin'),
                'promisor_false': ('remote.origin.promisor', 'false'),
            }
            if case in partial_keys:
                git('config', *partial_keys[case])
            if case == 'partial_include':
                included = repo / 'included.gitconfig'
                included.write_text('[remote "origin"]\n    partialCloneFilter = blob:none\n')
                git('config', 'include.path', str(included))
            if case == 'shallow':
                (repo / '.git/shallow').write_bytes(git('rev-parse', 'HEAD'))
            before = [p.read_bytes() for p in wrappers]
            rejected = case in ('dirty_edit', 'committed_edit', 'body_edit', 'wrong_target', 'no_git', 'missing_baseline', 'missing_historical_source', 'shallow', 'missing_object', 'missing_object_plain', 'partial_filter', 'partial_extension', 'partial_include', 'promisor_false')
            if case == 'stable':
                with mock.patch.object(module.subprocess, 'run', side_effect=AssertionError('stable wrappers must not invoke Git')):
                    self.assertEqual(generate(clean), 0)
            elif case in ('missing_object', 'missing_object_plain'):
                trace = repo / 'git-trace.txt'
                with mock.patch.dict(os.environ, {'GIT_TRACE': str(trace)}):
                    self.assertEqual(generate(clean), 2)
                self.assertNotIn('fetch', trace.read_text())
                if case != 'missing_object_plain':
                    self.assertNotIn('ls-tree', trace.read_text())
                    self.assertNotIn(' show ', trace.read_text())
            else:
                self.assertEqual(generate(clean), 2 if rejected else 0)
            self.assertEqual(asset.read_text(), 'User asset remains')
            if rejected:
                self.assertEqual([p.read_bytes() for p in wrappers], before)
            else:
                for host, wrapper in zip(('claude', 'codex'), wrappers):
                    if variant == 'simple':
                        self.assertEqual(module.parse_frontmatter(wrapper.read_text()), module.parse_frontmatter(source.read_text()))
                    else:
                        canonical, _ = module.parse_frontmatter(source.read_text())
                        actual, _ = module.parse_frontmatter(wrapper.read_text())
                        self.assertEqual(actual, module.expected_frontmatter(source, canonical, host))
            # Running again must stay stable, including preserved user edits.
            after = [p.read_bytes() for p in wrappers]
            self.assertEqual(generate(clean), 2 if rejected else 0)
            self.assertEqual([p.read_bytes() for p in wrappers], after)

    def test_matrix(self):
        cases = ('stable', 'canonical_dirty', 'canonical_committed', 'override', 'existing_override', 'dirty_edit', 'committed_edit', 'missing_baseline', 'no_git', 'legacy', 'body_edit', 'wrong_target', 'missing_historical_source', 'shallow', 'missing_object', 'missing_object_plain', 'partial_filter', 'partial_extension', 'partial_include', 'promisor_false')
        variants = ('legacy', 'modern', 'simple')
        if GENERATOR.is_file():
            text = GENERATOR.read_text()
            variants = ('simple' if 'def expected_frontmatter(' not in text else 'modern' if 'HOST_EXPOSED_SKILLS' in text else 'legacy',)
        for variant in variants:
            for case in cases:
                if variant == 'simple' and case in ('override', 'existing_override'):
                    continue
                for clean in (False, True):
                    with self.subTest(variant=variant, case=case, clean=clean):
                        self.run_case(variant, case, clean)

if __name__ == '__main__':
    unittest.main()
