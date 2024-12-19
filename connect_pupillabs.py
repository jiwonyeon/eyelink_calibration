""" 
    connect to the pupil labs glasses using IP
    
    ### common trouble shooting
    1. check whether pupillabs realtime api is installed (https://github.com/pupil-labs/realtime-network-api?tab=readme-ov-file)
    2. check whether the pupillab's phone device is connected to the same network as the computer
    3. check whether your computer device is not connected to a VPN
    4. if the connection doesn't work, try restart the phone device
    
    ### usage
    - To check the connection within this code, uncomment the line 19 and change the ip address
    - Otherwise, call the connect_pupillabs function in a script 
    
    jiwon yeon, 2024
"""

from pupil_labs.realtime_api.simple import Device

def connect_device(ip):
    device = Device(address=ip, port="8080")

    print(f"Phone IP address: {device.phone_ip}")
    print(f"Phone name: {device.phone_name}")
    print(f"Battery level: {device.battery_level_percent}%")
    print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
    print(f"Serial number of connected glasses: {device.serial_number_glasses}")
    
    return device

# Only execute this when running this script directly
if __name__ == "__main__":
    ip = "10.35.125.124"  # Hardcoded IP for standalone testing
    device = connect_device(ip)