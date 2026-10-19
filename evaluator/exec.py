import ast
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap


def extract_code(response):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', response, re.I | re.S)
    candidate = max(blocks, key=len) if blocks else response
    return re.sub(r"^\s*Here(?:'s| is)\s+(?:the\s+)?(?:code|implementation):\s*", '', candidate, flags=re.I).strip('\n')


class CodeRunner:
    def __init__(self, python=None, timeout=10, allow_unsandboxed=False):
        self.python = python or sys.executable
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Code timeout must be finite and positive')
        self.timeout = timeout
        self.allow_unsandboxed = allow_unsandboxed
        self.specs = json.loads(Path(__file__).with_name('cases.json').read_text())

    def run(self, record, response):
        spec = self.specs.get(record['source_id'])
        if not spec:
            return {'determinate': False, 'error': 'benchmark_test_not_found'}
        candidate = extract_code(response)
        if not candidate.strip():
            return {'determinate': True, 'passed': False, 'error': 'empty_code'}

        try:
            parsed = ast.parse(textwrap.dedent(candidate))
            has_entry = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == spec['entry_point'] for node in parsed.body)
        except SyntaxError:
            has_entry = False
        if not has_entry:
            candidate = spec['prompt'] + textwrap.indent(textwrap.dedent(candidate), '    ')
        else:
            candidate = textwrap.dedent(candidate)
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            return {'determinate': True, 'passed': False, 'error': f'SyntaxError: {exc.msg}'}
        ending = (f"\ncheck({spec['entry_point']})\n" if spec['kind'] == 'humaneval'
                  else "\nif __name__ == '__main__':\n    unittest.main()\n")
        program = candidate + '\n' + spec['test'] + ending
        return self._execute(program)

    def _execute(self, program):
        sandbox = shutil.which('sandbox-exec')
        if not sandbox and not self.allow_unsandboxed:
            return {'determinate': False, 'error': 'sandbox_unavailable; use allow_unsandboxed_code=True only in an isolated container/host'}
        with tempfile.TemporaryDirectory(prefix='utilitybench-code-') as temp:
            path = Path(temp)
            script = path / 'test.py'
            script.write_text(program)
            cmd = [self.python, '-I', str(script)]
            if sandbox and not self.allow_unsandboxed:

                profile = ('(version 1)(deny default)(allow process*)(allow file-read*)(allow sysctl-read)'
                           f'(allow file-write* (subpath {json.dumps(str(path.resolve()))}))'
                           '(deny network*)')
                cmd = [sandbox, '-p', profile, *cmd]
            env = {'PATH': os.environ.get('PATH', ''), 'PYTHONHASHSEED': '0',
                   'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': temp,
                   'MPLCONFIGDIR': temp, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

            def limits():
                import resource
                for name, value in [('RLIMIT_CPU', max(1, int(self.timeout))),
                                    ('RLIMIT_FSIZE', 16 * 1024 * 1024),
                                    ('RLIMIT_AS', 2 * 1024 * 1024 * 1024)]:
                    if name == 'RLIMIT_AS' and sys.platform == 'darwin':
                        continue
                    if not hasattr(resource, name):
                        continue
                    key = getattr(resource, name)
                    try:
                        hard = resource.getrlimit(key)[1]
                        bound = value if hard == resource.RLIM_INFINITY else min(value, hard)
                        resource.setrlimit(key, (bound, bound))
                    except (OSError, ValueError):
                        pass

            log = path / 'output.log'
            with log.open('wb') as stream:
                try:
                    process = subprocess.Popen(cmd, cwd=temp, env=env, stdout=stream, stderr=stream,
                                               start_new_session=True,
                                               preexec_fn=limits if os.name == 'posix' else None)
                except OSError as exc:
                    return {'determinate': False, 'error': f'code_runtime_unavailable: {exc}'}
                timed_out = False
                try:
                    process.wait(timeout=self.timeout + 1)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    if os.name == 'posix':
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif timed_out:
                        process.kill()
                    process.wait()
            if timed_out:
                return {'determinate': True, 'passed': False, 'error': 'timeout'}
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 3000))
                output = stream.read().decode('utf-8', errors='replace')
            if 'sandbox_apply:' in output or re.search(r'(?:ModuleNotFoundError|ImportError|PermissionError):', output):
                return {'determinate': False, 'error': output.strip()}
            return {'determinate': True, 'passed': process.returncode == 0,
                    'error': None if process.returncode == 0 else output.strip() or f'exit_code={process.returncode}'}
