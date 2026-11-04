# QT3 testing
1. Make sure RFSoC4x2 board is on. It should display the `<RFSoC_IP>` on the led screen.
2. Make sure the oscilloscope is on, then connect to it @ `128.95.31.216/Instrument/novnc/vnc_auto.php`
3. Connect to RFSoC4x2 and start Pyro server @ `<RFSoC_IP>`
4. Navigate to `/qick-dawg/run_server` and:
- Click on `name_server.ipynb`, Run every cell.
- Click on `qick_daemon.ipynb`, Run every cell.
5. Run tests //TODO