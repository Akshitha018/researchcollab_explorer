#!/usr/bin/env python3
"""
Launcher script for ResearchCollab Explorer Streamlit dev server.
Translates CLI arguments from npm/Vite runners (such as --port, --host)
into native Streamlit arguments (--server.port, --server.address).
"""

import sys
import os

def main():
    port = "3000"
    host = "0.0.0.0"
    
    # Parse CLI flags that might be forwarded by npm/Vite runners
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--port" and i + 1 < len(args):
            port = args[i + 1]
            i += 2
        elif arg.startswith("--port="):
            port = arg.split("=", 1)[1]
            i += 1
        elif arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
            i += 1
        else:
            i += 1

    streamlit_cmd = [
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false"
    ]

    print(f"Launching Streamlit server on {host}:{port}...", flush=True)
    os.execvp("streamlit", streamlit_cmd)

if __name__ == "__main__":
    main()
