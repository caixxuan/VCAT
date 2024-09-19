# launcher.py
# 程序崩溃后可自动重启！数据会被缓存记录！
# 2023/07/07

import subprocess
import sys
import time

script = 'SAC.py'
def restart_program():
    """Restart the target script."""
    subprocess.Popen([sys.executable, '-u', script])

if __name__ == '__main__':
    while True:
        print("Starting " + script)
        process = subprocess.Popen([sys.executable, '-u', script])
        process.wait()  # Wait for the target script to finish
        if process.returncode != 0:  # Check if the script terminated abnormally
            print("Target script terminated unexpectedly, restarting...")
            restart_program()
        else:
            print("Target script finished normally.")
            break  # Exit the loop if the script finished normally
        time.sleep(1)  # Add a short delay before restarting
