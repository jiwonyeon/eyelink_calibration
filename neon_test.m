global device

ip = '10.35.125.54';
device = pyrunfile("connect_pupillabs.py","device",ip=ip);
if isempty(device) || isempty(device.phone_id)
    error("Pupil lab device is not connected.")
else
    print(device)
end

