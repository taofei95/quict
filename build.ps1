$ErrorActionPreference = "Stop"

function Test-Command {
    param (
        [Parameter(Position = 0)]
        [string]
        $Command
    )
    try {
        Write-Host "Detecting $Command ..."
        $FullCommand = (Get-Command $Command).Source
        Write-Host "Found $Command($FullCommand)`n"
        return $FullCommand
    }
    catch {
        Write-Error "Cannot find $Command"
    }    
}

function Use-VS {
    $VsWhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VsInstallDir = & $VsWhere -latest -property installationPath
    Import-Module "$VsInstallDir\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $VsInstallDir -SkipAutomaticLocation -Arch amd64 -HostArch amd64
}

# Detect Visual Studio installation

Write-Host "Detecting Visual Studio ..."
Use-VS
Write-Host "`n"

# Verify CLI tools.

$Powershell = Test-Command "powershell.exe"

$Python3 = Test-Command "python3.exe"

Test-Command "ninja.exe"

$ClangCl = Test-Command "clang-cl.exe"

# Setup build environment.

$ENV:CMAKE_GENERATOR = "Ninja"
$ENV:CC = $ClangCl
$ENV:CXX = $ClangCl
$OldComSpec = $ENV:ComSpec
$ENV:ComSpec = $Powershell

# Invoke build.

try {
    Write-Host "Build python wheel."
    & $Python3 .\setup.py bdist_wheel
}
finally {
    $ENV:ComSpec = $OldComSpec
}

