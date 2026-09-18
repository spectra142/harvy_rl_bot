#!/usr/bin/env python3
"""Run a command on the remote Minecraft server host via SSH.

Usage:
    MC_HOST_IP=192.168.1.10 MC_SSH_USER=spectra MC_SSH_PASS=... \
        python scripts/mc_ssh.py 'systemctl status minecraft' [timeout_seconds]

Requires: pip install paramiko
"""
import os
import sys

try:
    import paramiko
except ImportError:
    sys.exit("paramiko not installed: pip install paramiko")

host = os.environ.get("MC_HOST_IP", "192.168.18.19")
user = os.environ.get("MC_SSH_USER", "spectra")
password = os.environ.get("MC_SSH_PASS")
if not password:
    sys.exit("Set MC_SSH_PASS in the environment (never hardcode it).")

cmd = sys.argv[1] if len(sys.argv) > 1 else "echo no-command"
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password, timeout=10)
stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
out = stdout.read().decode("utf-8", errors="replace")
err = stderr.read().decode("utf-8", errors="replace")
rc = stdout.channel.recv_exit_status()
sys.stdout.write(out)
if err.strip():
    sys.stderr.write(err)
client.close()
sys.exit(rc)
