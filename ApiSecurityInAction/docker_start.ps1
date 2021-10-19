

if ($env:CHAPTER == $null) {
    Write-Host "No chapter selected. Ensure you pass the `CHAPTER` environment variable with one of the following options.`n"
    Break
}

if ($chapters -notcontains $env:CHAPTER) {
    Write-Host "Invalid chapter ${$env:CHAPTER} selected. Select from one of the following.`n${chapters -join ", "}"
    Break
}


$server = "java -cp ./h2/bin/h2.jar org.h2.tools.Server"
$runscript = "java -cp ./h2/bin/h2.jar org.h2.tools.RunScript"

