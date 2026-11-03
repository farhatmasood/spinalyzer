"""Start the local server: ``python -m spinalyzer``."""
import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="spinalyzer")
    parser.add_argument("--host", default=os.environ.get("SPINALYZER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SPINALYZER_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run("spinalyzer.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
