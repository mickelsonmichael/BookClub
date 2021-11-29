using System.ComponentModel.DataAnnotations;

namespace NatterApi.Options
{
    public class KeycloakOptions
    {
        public const string ConfigKey = "Keycloak";

        [Required(AllowEmptyStrings = false)]
        public string ClientId { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string Secret { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string IntrospectionEndpoint { get; set; }
        [Required(AllowEmptyStrings = false)]
        public string RevokeEndpoint { get; set; }
    }
}
