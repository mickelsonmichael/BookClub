$user = "admin"
$pass = "admin"
$container = "api-security-as"

Write-Host "Starting keycloak server with credentials $user`:$pass."

try {
    Write-Debug "Removing existing instance."
    docker stop $container | Out-Null
    docker rm $container | Out-Null
}
catch {
    Write-Debug "Starting fresh instance."
}

docker run `
    -d `
    --name $container `
    -p 8080:8080 `
    -e KEYCLOAK_USER=$user `
    -e KEYCLOAK_PASSWORD=$pass `
    -e DB_VENDOR=h2 `
    jboss/keycloak `
    | Out-Null

$url = "http://localhost:8080/auth";

Write-Host "Waiting for server to be up and ready..." -NoNewline

$healthCheck = $null
while ($null -eq $healthCheck || $healthCheck.StatusCode -ne 200) {
    try {
        $healthCheck = Invoke-WebRequest -Method HEAD $url
    }
    catch {
        Start-Sleep 2
    }
}

Write-Host "Ready!"

Write-Host "Getting access token for API..." -NoNewline
$tokenResp = (Invoke-WebRequest -Uri "$($url)/realms/master/protocol/openid-connect/token" `
    -Method POST `
    -ContentType "application/x-www-form-urlencoded" `
    -Body 'grant_type=password&client_id=admin-cli&username=admin&password=admin').Content `
    | ConvertFrom-Json | Select-Object -ExpandProperty access_token
Write-Host "Got it!"

$headers = @{
    Authorization = "Bearer $($tokenResp)"
}

Write-Host "Creating realm 'api-security'..." -NoNewline
Invoke-WebRequest -Uri "$url/admin/realms" `
    -Method POST `
    -ContentType "application/json" `
    -Headers $headers `
    -Body '
    {
        "realm": "api-security"
    }' | Out-Null
Write-Host "Created!"

Write-Host "Adding Natter to list of realm's clients and creating json file..." -NoNewline
$clientUrl = (Invoke-WebRequest -Uri "$url/admin/realms/api-security/clients" `
    -Method POST `
    -ContentType "application/json" `
    -Headers $headers `
    -Body '
    {
        "rootUrl": "https://localhost:4567",
        "protocol": "openid-connect",
        "clientId": "natter"
    }
    ').Headers["Location"]

(Invoke-WebRequest -Uri "$clientUrl/installation/providers/keycloak-oidc-keycloak-json" `
    -Headers $headers `
    ).Content | Out-File -FilePath "..\dotnet\NatterApi\keycloak-oidc.json"
Write-Host "Added!"




# Start-Process "$url/admin"
