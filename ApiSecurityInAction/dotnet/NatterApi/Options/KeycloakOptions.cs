using System.ComponentModel.DataAnnotations;

namespace NatterApi.Options
{
    public class KeycloakOptions
    {
        public const string ConfigKey = "Keycloak";

#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
        [Required(AllowEmptyStrings = false)]
        public string ClientId { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string Secret { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string IntrospectionEndpoint { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string RevokeEndpoint { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string JwkSetEndpoint { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string Audience { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string Issuer { get; set; }
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
    }
}
