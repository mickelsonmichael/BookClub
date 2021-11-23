$token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJBWGlYeHNROEQ4UDdMbHJTeG1GTE9xdGUweG89Iiwic3ViIjoiZGVtbyIsImF1ZCI6Imh0dHBzOi8vbG9jYWxob3N0OjQ1NjciLCJleHAiOjE2Mzc2OTAxOTIuMH0.IxoY7iJ7BW4x6M9xjfAoaAVX4g_eywx2vw8uCumP0Tk"

$Headers = @{
    Authorization = "Bearer $($token)"
}

$loginResp = Invoke-WebRequest -Method GET -ContentType "application/json" -Uri https://localhost:4567/sessions -Headers $Headers

Write-Host $($loginResp.StatusCode)
