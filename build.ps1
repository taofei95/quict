function Test-Command {
    param (
        [Parameter(Position = 0)]
        [string]
        $Command
    )
    $ErrorActionPreference = "Stop"
    try {
        Write-Host "Detecting $Command ..."
        $FullCommand = (Get-Command $Command).Source
        return $FullCommand
    }
    catch {
        Write-Error "Cannot find $Command"
    }    
}

function Use-VS
{
    $VsWhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $VsInstallDir = & $VsWhere -latest -property installationPath
    Import-Module "$VsInstallDir\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $VsInstallDir -SkipAutomaticLocation -Arch amd64 -HostArch amd64
}

# Detect Visual Studio installation

Use-VS

# Verify CLI tools.

$Python3 = Test-Command "python3.exe"
Write-Host "Found python3($Python3)`n"

$Ninja = Test-Command "ninja.exe"
Write-Host "Found ninja($Ninja)`n"

$ClangCl = Test-Command "clang-cl.exe"
Write-Host "Found clang-cl($ClangCl)`n"

# Setup build environment.

$ENV:CMAKE_GENERATOR="Ninja"
$ENV:CC=$ClangCl
$ENV:CXX=$ClangCl
$ENV:ComSpec="powershell.exe"

# Invoke build.

Write-Host "Build python wheel."
& $Python3 .\setup.py bdist_wheel

