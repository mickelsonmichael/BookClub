$token = "token"

$Headers = @{
    Authorization = "Bearer $($token)"
}

$loginResp = Invoke-WebRequest -Method GET -ContentType "application/json" -Uri https://localhost:4567/sessions -Headers $Headers

Write-Host $($loginResp.StatusCode)
