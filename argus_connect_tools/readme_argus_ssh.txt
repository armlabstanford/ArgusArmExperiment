ARGUS SSH INFO:

To connect via the hotspot:

ssh argus@192.168.50.10

And enter password: ArmLab19
(password shouldn't be needed because this is a known host, but still)

for ethernet if connected:

ssh argus@10.42.0.59

If the ip has changed for wifi, run this on the rog:
sudo arp-scan --interface=wlxc83a35c807b4 --localnet

for ethernet:
sudo arp-scan --localnet

they'll check for local devices, you should see a device with:
ip_addr     mac_addr     Raspberry Pi Trading Ltd


use ssh argus@ip_addr to login

