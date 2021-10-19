$chapters = {
    "chapter01", "chapter02", "chapter02-end", "chapter03", "chapter03-end", "chapter04", "chapter04-end", "chapter05", "chapter05-end", "chapter06", "chapter06-end", "chapter07", "chapter07-end", "chapter08", "chapter08-end", "chapter09", "chapter09-end", "chapter10", "chapter10-end", "chapter11", "chapter11-end", "chapter12", "chapter12-end", "chapter13", "chapter13-end", "database_encryption", "feature/oauth2"
};

if ($env:CHAPTER == $null) {
    Write-Host "No chapter selected. Ensure you pass the `CHAPTER` environment variable with one of the following options.`n"
    Break
}

if ($chapters -notcontains $env:CHAPTER) {
    Write-Host "Invalid chapter ${$env:CHAPTER} selected. Select from one of the following.`n${chapters -join ", "}"
    Break
}

if (docker ps)

docker run maven:3-openjdk-8

Invoke-WebRequest -Uri https://github.com/NeilMadden/apisecurityinaction/archive/refs/heads/master.zip -OutFile C:\src\source.zip

$server = "java -cp ./h2/bin/h2.jar org.h2.tools.Server"
$runscript = "java -cp ./h2/bin/h2.jar org.h2.tools.RunScript"

