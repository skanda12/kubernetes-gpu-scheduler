# Run in an ELEVATED PowerShell on each Windows laptop (mirrored networking option).
# Opens the ports k3s needs between laptops on the same Wi-Fi / hotspot. Remove with -Remove.
param([switch]$Remove)
$rules = @(
  @{Name="gpuorch-k3s-api";      Proto="TCP"; Port="6443"},
  @{Name="gpuorch-kubelet";      Proto="TCP"; Port="10250"},
  @{Name="gpuorch-flannel-vxlan";Proto="UDP"; Port="8472"},
  @{Name="gpuorch-gpu-agent";    Proto="TCP"; Port="9400"},
  @{Name="gpuorch-nodeports";    Proto="TCP"; Port="30000-32767"}
)
foreach ($r in $rules) {
  if ($Remove) {
    Remove-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
  } else {
    New-NetFirewallRule -DisplayName $r.Name -Direction Inbound -Protocol $r.Proto -LocalPort $r.Port -Action Allow -Profile Private | Out-Null
  }
}
# Hyper-V firewall (WSL2 mirrored mode) - allow inbound to the WSL VM
if (-not $Remove) {
  Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow -ErrorAction SilentlyContinue
}
Write-Host ($(if ($Remove) {"removed"} else {"added"}) + " gpuorch firewall rules")
