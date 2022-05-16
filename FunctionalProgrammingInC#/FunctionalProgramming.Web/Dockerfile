FROM mcr.microsoft.com/dotnet/sdk:6.0 as sdk

WORKDIR /build

COPY *.csproj .

RUN dotnet restore

COPY . .

RUN dotnet publish --configuration Release --output /publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:6.0 as runtime

WORKDIR /app

COPY --from=sdk /publish .

ENTRYPOINT [ "dotnet", "FunctionalProgramming.Web.dll" ]
