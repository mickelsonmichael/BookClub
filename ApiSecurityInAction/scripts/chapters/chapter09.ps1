$creds="Basic $([Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("demo:password")))"
$creds2="Basic $([Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("demo2:password")))"

Write-Host "Registering demo..." -NoNewline
Invoke-WebRequest -SkipCertificateCheck `
    -Uri https://localhost:4567/users `
    -Method POST `
    -ContentType "application/json" `
    -Body '{ "username": "demo", "password": "password" }' `
    | Select-Object -ExpandProperty StatusCode

Write-Host "Registering demo2..." -NoNewline
Invoke-WebRequest -SkipCertificateCheck `
    -Uri https://localhost:4567/users `
    -Method POST `
    -ContentType "application/json" `
    -Body '{ "username": "demo2", "password": "password" }' `
    | Select-Object -ExpandProperty StatusCode

Write-Host "Creating space with demo 1..."
$url = Invoke-WebRequest -SkipCertificateCheck `
    -Uri https://localhost:4567/spaces `
    -Headers @{
        Authorization = $creds
    } `
    -Method POST `
    -ContentType "application/json" `
    -Body "{`"name`":`"the space`", `"owner`": `"demo`"}" `
    | Select-Object -ExpandProperty Content `
    | ConvertFrom-Json `
    | Select-Object -ExpandProperty messages_rw

Write-Host "Attempting to use capability URL as demo2"
Invoke-WebRequest -SkipCertificateCheck `
    -Uri $url `
    -Headers @{ Authorization = $creds2 }`
    | Select-Object -ExpandProperty Content

