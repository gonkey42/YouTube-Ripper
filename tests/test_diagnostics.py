import subprocess
import unittest
from unittest.mock import patch

import diagnostics


class DependencyDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.access = patch("diagnostics.os.access", return_value=False).start()
        self.which = patch("diagnostics.shutil.which", side_effect=lambda name: f"/bin/{name}").start()
        self.run = patch("diagnostics.subprocess.run", side_effect=self.successful_version).start()
        self.addCleanup(patch.stopall)

    @staticmethod
    def successful_version(command, **kwargs):
        output = "ffmpeg version 8.0\n" if command[1] == "-version" else "v22.1.0\n"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def test_healthy_dependencies(self):
        self.assertEqual(diagnostics.dependency_issues(), [])
        self.assertEqual(self.run.call_count, 2)
        for call in self.run.call_args_list:
            self.assertEqual(call.kwargs["timeout"], diagnostics.CHECK_TIMEOUT_SECONDS)

    def test_reports_missing_dependencies_with_install_instructions(self):
        self.which.side_effect = None
        self.which.return_value = None
        issues = diagnostics.dependency_issues()
        self.assertEqual(len(issues), 2)
        self.assertIn("brew install ffmpeg", issues[0])
        self.assertIn("Node.js 22", issues[1])
        self.assertIn("brew install node", issues[1])
        self.run.assert_not_called()

    def test_can_check_node_without_ffmpeg(self):
        self.assertEqual(diagnostics.dependency_issues(require_ffmpeg=False), [])
        self.which.assert_called_once_with("node")
        self.run.assert_called_once()

    def test_prefers_homebrew_node_to_path(self):
        self.access.return_value = True
        self.assertEqual(diagnostics.dependency_issues(require_ffmpeg=False), [])
        self.which.assert_not_called()
        self.assertEqual(self.run.call_args.args[0], [str(diagnostics.HOMEBREW_NODE_PATH), "--version"])

    def test_accepts_current_and_newer_node(self):
        for version in ("v22.0.0", "v24.4.1", "v26.0.0"):
            with self.subTest(version=version):
                self.run.side_effect = None
                self.run.return_value = subprocess.CompletedProcess([], 0, stdout=version)
                self.assertEqual(diagnostics.dependency_issues(require_ffmpeg=False), [])

    def test_rejects_old_and_unrecognized_node_versions(self):
        for version, message in (("v20.19.0", "too old"), ("v21.7.3", "too old"), ("unknown", "unrecognized"), ("", "unrecognized")):
            with self.subTest(version=version):
                self.run.side_effect = None
                self.run.return_value = subprocess.CompletedProcess([], 0, stdout=version)
                issues = diagnostics.dependency_issues(require_ffmpeg=False)
                self.assertEqual(len(issues), 1)
                self.assertIn(message, issues[0])
                self.assertIn("22 or newer", issues[0])

    def test_reports_unrecognized_ffmpeg(self):
        def output(command, **kwargs):
            result = self.successful_version(command)
            if command[1] == "-version":
                result.stdout = "some other program"
            return result
        self.run.side_effect = output
        issues = diagnostics.dependency_issues()
        self.assertEqual(len(issues), 1)
        self.assertIn("ffmpeg returned an unrecognized version", issues[0])

    def test_reports_nonzero_exits(self):
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 127, stdout="", stderr="broken")
        issues = diagnostics.dependency_issues()
        self.assertEqual(len(issues), 2)
        self.assertTrue(all("exit 127" in issue for issue in issues))

    def test_reports_unrunnable_executables(self):
        for error in (FileNotFoundError(), PermissionError(), OSError("invalid binary")):
            with self.subTest(error=error):
                self.run.side_effect = error
                issues = diagnostics.dependency_issues()
                self.assertEqual(len(issues), 2)
                self.assertTrue(all("could not run" in issue for issue in issues))

    def test_reports_timeouts(self):
        self.run.side_effect = subprocess.TimeoutExpired("version", diagnostics.CHECK_TIMEOUT_SECONDS)
        issues = diagnostics.dependency_issues()
        self.assertEqual(len(issues), 2)
        self.assertTrue(all("did not respond" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
