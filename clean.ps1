function Print-With-Color {
    param(
        [Parameter(Position = 0)]
        [string]
        $Color,
        [Parameter(Position = 1)]
        [string]
        $Content
    )
    Write-Host -ForeGroundColor $Color $Content
}

function Print-Segment {
    Print-With-Color -Color Green -Content "================================================================================"
}

function Print-Cyan {
    param(
        [Parameter(Position = 0)]
        [string]
        $Content
    )
    Print-With-Color -Color Cyan -Content $Content
}

function Print-Magenta {
    param(
        [Parameter(Position = 0)]
        [string]
        $Content
    )
    Print-With-Color -Color Magenta -Content $Content
}

$PrjBuildDir = "$PSScriptRoot\build"
$PrjEggDir = "$PSScriptRoot\QuICT.egg-info"
$PrjDistDir = "$PSScriptRoot\dist"

foreach ($dir in @($PrjBuildDir, $PrjEggDir, $PrjDistDir)) {
    if (Test-Path -Path $dir) {
        Write-Host "Deleting $dir ..."
        Remove-Item -Recurse $dir
    }
}

Write-Host "Remove all built .pyd files"

Get-ChildItem *.pyd -Recurse | ForEach-Object { Remove-Item -Path $_.FullName }

Print-Magenta "Cleaned."
