# MOBILE_ACCESS.md

## Phone access
Jarvis API runs locally. For secure access from phone:

- **Tailscale (recommended)**: Run server on laptop (0.0.0.0 or 127.0.0.1 behind tailnet), access via laptop's Tailscale IP. No public internet exposure.
- **Local Wi-Fi (same network)**: Use laptop LAN IP. Only when on trusted network.
- **SSH tunnel**: ssh -L 8123:127.0.0.1:8123 user@laptop; access http://127.0.0.1:8123 on phone.

Server binds 0.0.0.0 now for LAN access (trusted networks). Keep behind VPN/Tailscale.
