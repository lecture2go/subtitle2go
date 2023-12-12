import argparse
import sys


def import_module(module_name, min_version=None):
    try:
        print(f"Checking engine import {module_name}...", end="", flush=True)
        module = __import__(module_name)

        if min_version:
            module_version = getattr(module, '__version__', None)
            if module_version and module_version < min_version:
                raise ImportError(f"Minimum version {min_version} required, found {module_version}")

        print("\033[92m success!\033[0m")
        return True
    except ImportError as e:
        print(f"\033[91m error! {e}\033[0m")
        return False


def is_whisper_patched():
    print(f"Checking if engine Whisper is patched to allow for status updates...", end="", flush=True)
    try:
        from whisper import transcribe
        import inspect
        # Get the parameters of the transcribe function
        transcribe_params = inspect.signature(transcribe).parameters

        is_patched = 'status' in transcribe_params
        if is_patched:
            print("\033[92m success!\033[0m")
        else:
            print(f"\033[91m error! {e}\033[0m")
        # Check if the 'status' parameter is present
        return is_patched

    except Exception as e:
        print(f"\033[91m error! {e}\033[0m")
        # Handle any exceptions that may occur during inspection
        print(f"Error checking if Whisper is patched: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check the availability and version of ASR engines.")
    parser.add_argument("--with-error", action="store_true", help="Exit with an error code if import fails.")
    args = parser.parse_args()

    modules_to_check = [
        {"name": "kaldi_decoder", "min_version": None},
        {"name": "espnet_streaming_decoder", "min_version": "0.1"},
        {"name": "speechcatcher", "min_version": "0.3.1"},
        {"name": "speechcatcher_decoder", "min_version": None},
        {"name": "whisper_decoder", "min_version": None}
    ]

    for module_info in modules_to_check:
        module_name = module_info["name"]
        min_version = module_info["min_version"]

        if not import_module(module_name, min_version) and args.with_error:
            print(f"\033[91mExiting with error due to failed import of {module_name}\033[0m")
            sys.exit(-1)

    is_whisper_patched()

    print("Note: you not need all engines to run subtitle2go.py - only the engine(s) that you are planning on using.")

if __name__ == "__main__":
    main()
