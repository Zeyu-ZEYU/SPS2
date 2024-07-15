import subprocess
import sys


def run_cmd(cmd, failure_warning=True, return_output=True):
    try:
        if return_output:
            return subprocess.check_output(cmd, shell=True, universal_newlines=True).strip("\n")
        else:
            return subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        if failure_warning:
            print("WARNING:", e, file=sys.stderr)
