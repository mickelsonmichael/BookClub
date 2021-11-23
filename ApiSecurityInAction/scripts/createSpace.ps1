$token = "token"

$Headers = @{
    Authorization = "Bearer $($token)"
}

$loginResp = Invoke-WebRequest -Method POST -Body '{"name":"test_space","owner":"demo"}' -ContentType "application/json" -Uri https://localhost:4567/spaces -Headers $Headers

Write-Host $($loginResp.StatusCode)
