import subprocess

# Path to the local image
local_image = "/home/pi/pictures/picture.jpg"

# SCP command
command = f"scp {local_image} your_windows_username@your_computer_ip:C:/path/to/target/directory"

# Execute SCP command
try:
    subprocess.run(command, shell=True, check=True)
    print("File successfully transferred")
except subprocess.CalledProcessError:
    print("File transfer failed")
