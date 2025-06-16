import subprocess
from typing import Tuple

class BVBRCShellExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def run_query(self, query: str) -> Tuple[str, str]:
        process = subprocess.Popen(
            query,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stderr += '\nTimeoutExpired'
        return stdout, stderr 