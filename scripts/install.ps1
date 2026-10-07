param([Parameter(Mandatory=$true)][string]$Archive,[Parameter(Mandatory=$true)][string]$Prefix,[Parameter(Mandatory=$true)][string]$Sha256)
$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/install.py" $Archive $Prefix $Sha256
if ($LASTEXITCODE -ne 0) { throw 'HyperL installation failed' }
