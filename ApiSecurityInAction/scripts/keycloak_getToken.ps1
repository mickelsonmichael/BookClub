$creds = "Basic $([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("natter:getmestuff")))"

$data = "client_id=natter&grant_type=password&scope=create_space+post_message&username=demo&password=changeit";

$headers = @{
    Authorization = $creds
}

$loginResp = Invoke-WebRequest -Method POST `
    -Body $data `
    -ContentType "application/x-www-form-urlencoded" `
    -Headers $headers `
    -Uri http://localhost:8080/auth/realms/api-security/protocol/openid-connect/token

Write-Host $($loginResp | ConvertFrom-Json | Select-Object -ExpandProperty access_token)
