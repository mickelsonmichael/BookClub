$creds = "Basic $([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("demo:password")))"

$Headers = @{
    Authorization = $creds
}

$loginResp = Invoke-WebRequest -Method POST -Body '' -ContentType "application/json" -Uri https://localhost:4567/sessions -Headers $Headers

Write-Host $($loginResp.Content)
