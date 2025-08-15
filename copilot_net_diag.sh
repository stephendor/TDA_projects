#!/usr/bin/env bash
set -euo pipefail

LOG="${1:-$HOME/copilot_net_diag.log}"
echo "# Copilot network diag start: $(date --iso-8601=seconds)" | tee -a "$LOG"

# Ensure tools exist or degrade gracefully
have() { command -v "$1" >/dev/null 2>&1; }

# Background monitors (prefixed lines, line-buffered)
pids=()

if have nmcli; then
  stdbuf -oL nmcli monitor | sed -u 's/^/[NMCLI] /' >> "$LOG" 2>&1 &
  pids+=($!)
fi

stdbuf -oL ip monitor link address route | sed -u 's/^/[IPMON] /' >> "$LOG" 2>&1 &
pids+=($!)

# Kernel messages (driver resets, DHCP, etc.)
stdbuf -oL journalctl -k -f -n0 | sed -u 's/^/[KERN]  /' >> "$LOG" 2>&1 &
pids+=($!)

# On exit, stop background jobs
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  echo "# Copilot network diag end: $(date --iso-8601=seconds)" | tee -a "$LOG"
}
trap cleanup EXIT

# Helper: field extractor
field() { awk -v k="$1" '{for(i=1;i<=NF;i++) if($i==k){print $(i+1); exit}}'; }

# Main poller
while true; do
  ts=$(date --iso-8601=seconds)

  # Default route towards a stable anycast IP
  route_line=$(ip route get 1.1.1.1 2>/dev/null | sed -n '1p' || true)
  iface=$(printf "%s\n" "$route_line" | field dev)
  src=$(printf "%s\n" "$route_line" | field src)
  via=$(printf "%s\n" "$route_line" | field via)

  # Public IPs via DNS (no HTTP dependencies)
  pub4=""
  pub6=""
  if have dig; then
    pub4=$(dig +short myip.opendns.com @resolver1.opendns.com 2>/dev/null | head -n1 || true)
    pub6=$(dig -6 +short AAAA myip.opendns.com @resolver1.opendns.com 2>/dev/null | head -n1 || true)
  fi

  # Active connections (helps detect VPNs)
  vpn="none"
  if have nmcli; then
    vpn=$(nmcli -t -f NAME,TYPE,DEVICE connection show --active 2>/dev/null \
      | awk -F: '$2=="vpn"{print $1"@"$3}' | paste -sd, -)
    [ -z "$vpn" ] && vpn="none"
  fi

  # Current Wi-Fi AP (BSSID/SSID) if applicable
  ap="n/a"
  if have iw; then
    # pick first managed interface with a link
    iface_wifi=$(iw dev 2>/dev/null | awk '/Interface/ {print $2; exit}')
    if [ -n "${iface_wifi:-}" ]; then
      link=$(iw dev "$iface_wifi" link 2>/dev/null || true)
      ssid=$(printf "%s\n" "$link" | awk -F': ' '/SSID:/ {print $2}')
      bssid=$(printf "%s\n" "$link" | awk '/Connected to/ {print $3}')
      [ -n "$ssid$bssid" ] && ap="${ssid:-?}@${bssid:-?}"
    fi
  fi

  printf "[SAMPLE] %s route{dev=%s via=%s src=%s} pub{v4=%s v6=%s} vpn{%s} wifi{%s}\n" \
    "$ts" "${iface:-?}" "${via:-?}" "${src:-?}" "${pub4:-?}" "${pub6:--}" "$vpn" "$ap" \
    | tee -a "$LOG"

  sleep 2
done
