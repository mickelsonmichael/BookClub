$token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJjTlY5SnlyQ1JBVC9SUWY0Q2lrVXRjclNUODQ9Iiwic3ViIjoiZGVtbyIsImF1ZCI6Imh0dHBzOi8vbG9jYWxob3N0OjQ1NjciLCJleHAiOjE2Mzc2ODk3NDIuMH0.9WGjSg9-6TQ1tMROvrtu_JDeb_nb-A52DDjRty0pEUw"

$Headers = @{
    Authorization = "Bearer $($token)"
}

$loginResp = Invoke-WebRequest -Method POST -Body '{"name":"test_space","owner":"demo"}' -ContentType "application/json" -Uri https://localhost:4567/spaces -Headers $Headers

Write-Host $($loginResp.StatusCode)
