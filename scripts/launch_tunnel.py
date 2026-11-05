#!/usr/bin/env python3
"""
Follica Zero-Cost Mobile & Remote Testing Tunnel Launcher.

Provides instant, secure public HTTPS access to the local Follica FastAPI backend
for testing on physical smartphones without router port forwarding or paid subscriptions.
Uses Cloudflare Tunnel (cloudflared quick tunnels - 100% free, zero account required).
"""

import os
import shutil
import subprocess
import sys
import time

PORT = 8000

def check_cloudflared():
    """Checks whether cloudflared executable is available on PATH."""
    return shutil.which("cloudflared") is not None

def print_setup_instructions():
    print("=" * 70)
    print("Cloudflare Tunnel CLI (cloudflared) is not detected on your PATH.")
    print("To test on physical smartphones for free without port forwarding:")
    print("=" * 70)
    if sys.platform.startswith("win"):
        print("1. Download the free executable:")
        print("   winget install Cloudflare.cloudflared")
        print("   OR download from: https://github.com/cloudflare/cloudflared/releases")
    elif sys.platform == "darwin":
        print("1. Install via Homebrew:")
        print("   brew install cloudflared")
    else:
        print("1. Install via package manager:")
        print("   sudo apt-get install cloudflared")
    print("2. Run this script again, or run directly in your terminal:")
    print(f"   cloudflared tunnel --url http://127.0.0.1:{PORT}")
    print("=" * 70)

def main():
    print("[Follica] Initializing zero-cost public test tunnel...")
    if not check_cloudflared():
        print_setup_instructions()
        return

    print(f"[Follica] Starting quick tunnel pointing to http://127.0.0.1:{PORT}...")
    try:
        cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            if "trycloudflare.com" in line:
                print("\n" + "=" * 70)
                print("[SUCCESS] Your Follica Mobile Test Tunnel is live!")
                print("Open the trycloudflare.com URL above on your phone browser or Expo app.")
                print("=" * 70 + "\n")
    except KeyboardInterrupt:
        print("\n[Follica] Shutting down tunnel.")

if __name__ == "__main__":
    main()
