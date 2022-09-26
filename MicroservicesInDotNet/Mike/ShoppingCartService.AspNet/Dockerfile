FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build

WORKDIR /src

COPY ShoppingCartService.csproj .

RUN dotnet restore

COPY . .

RUN dotnet build -c Release -o /build

FROM build as publish

RUN dotnet publish -c Release -o -o /publish

FROM mcr.micrososft.com/dotnet/aspnet:6.0 as runtime

WORKDIR /app

EXPOSE 80

COPY --from=publish /publish .

ENTRYPOINT ["dotnet", "ShoppingCart.dll"]
