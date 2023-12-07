import argparse
import sys

def import_module(module_name):
    try:
        print(f"Checking engine import {module_name}...", end="", flush=True)
        __import__(module_name)
        print("\033[92m success!\033[0m")
        return True
    except ImportError:
        print("\033[91m error!\033[0m")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check the availability of ASR engines.")
    parser.add_argument("--with-error", action="store_true", help="Exit with an error code if import fails.")
    args = parser.parse_args()

    modules_to_check = ["kaldi_decoder", "speechcatcher_decoder", "whisper_decoder"]

    for module_name in modules_to_check:
        if not import_module(module_name) and args.with_error:
            print(f"\033[91mExiting with error due to failed import of {module_name}\033[0m")
            sys.exit(-1)

if __name__ == "__main__":
    main()