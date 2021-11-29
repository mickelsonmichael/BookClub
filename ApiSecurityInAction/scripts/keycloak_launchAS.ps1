$user = "admin"
$pass = "admin"
$container = "api-security-as"

$keyCloakSettings = Get-Content -Path "../dotnet/NatterApi/appsettings.json" | ConvertFrom-Json | Select-Object -ExpandProperty Keycloak
$clientId = $keyCloakSettings.ClientId
$clientSecret = $keyCloakSettings.Secret

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
        "realm": "api-security",
        "enabled": true
    }' | Out-Null
Write-Host "Created!"

$scopes = "full_access", "create_space", "post_message", "read_message", "list_message", "add_member", "delete_message"

Write-Host "Adding scopes to realm..." -NoNewline

foreach ($scope in $scopes)
{
    Invoke-WebRequest -Uri "$url/admin/realms/api-security/client-scopes" `
        -Method POST `
        -ContentType "application/json" `
        -Headers $headers `
        -Body "
        {
            `"name`": `"$scope`",
            `"protocol`": `"openid-connect`"
        }
        " | Out-Null
}
Write-Host "All $($scopes.Length) scopes added!"

Write-Host "Adding Natter to list of realm's clients and creating json file..." -NoNewline
$clientUrl = (Invoke-WebRequest -Uri "$url/admin/realms/api-security/clients" `
    -Method POST `
    -ContentType "application/json" `
    -Headers $headers `
    -Body "
    {
        `"rootUrl`": `"https://localhost:4567`",
        `"protocol`": `"openid-connect`",
        `"clientId`": `"$clientId`",
        `"secret`": `"$clientSecret`",
        `"publicClient`": false,
        `"optionalClientScopes`": [$($scopes | Join-String -Separator "," -DoubleQuote)]
    }
    ").Headers["Location"]

(Invoke-WebRequest -Uri "$clientUrl/installation/providers/keycloak-oidc-keycloak-json" `
    -Headers $headers `
    ).Content | Out-File -FilePath "..\dotnet\NatterApi\keycloak-oidc.json"
Write-Host "Added!"

Write-Host "!!! Access the client endpoints using `"$clientId`:$clientSecret`"."

Write-Host "Creating user `"demo`" with password `"changeit`"..." -NoNewline
Invoke-WebRequest -Method POST `
    -ContentType "application/json" `
    -Headers $headers `
    -Uri "$url/admin/realms/api-security/users" `
    -Body '
        {
            "enabled": true,
            "username": "demo",
            "credentials": [{
                "type": "password",
                "value": "changeit",
                "temporary": false
            }]
        }
    ' | Out-Null
Write-Host "It's alive!"


Start-Process "$url/admin"
