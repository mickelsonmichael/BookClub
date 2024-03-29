# Capability-based Security and Macaroons

- [Michael's Notes](#michael)

## Michael

While identity-based access control is great, it doesn't really mesh well with sharing. If a user wants to show a message in Natter to another user, they'd either have to grant that user permissions, which is cumbersome and exposes potential vulnerabilities, or find some workaround to get their message shared.

Capability-based security seeks to solve that problem, and to rectify ["confused deputy"](https://en.wikipedia.org/wiki/Confused_deputy_problem) attacks. A CSRF attack is an example of a confused deputy attack, since the attack is utilizing a patsy with greater access to perform some action they'd normally not be allowed to do.

### Capability-based security (Section 9.1)

The exact definition of capability-based security can be a little vague and examples given are often in terms of OS security, rather than APIs (although Madden gives an example in section 9.2). But at its core, a capability is "an unforgeable reference to an object or resource together with a set of permissions to access that resource" (Madden, 2020). Or, more simply, it is an identifier for an object, such as an ID, and a set of permissions a user with this capability would have access to.

Madden provides an example of a UNIX filesystem, which I won't repeat here, and then lists a couple advantages and disadvantages to capabilities verses identity-based access control.

1. Capabilities can not be forged, and you cannot send a request for a resource you can't access. Meanwhile in an identity-based approach, anyone can request the action and be either approved or denied
2. Capabilities are more fine-grained, and permissions can be granted in smaller, more *principle of least privilege*-friendly way
3. It is harder to list who has access to what resources with a capability-based system. You can determine that someone shared a capability, but you cannot easily determine who they shared with with
4. Revoking access using a capability-based system is more complicated and not always possible. It also may be too broad in scope and revocations may result in too many capabilities being removed.

Check out the article [Capability Myths Demolished](https://srl.cs.jhu.edu/pubs/SRL2003-02.pdf) (M.S. Miller, K. Yee, J. Shapiro, 2003) for some assurances that capability-based systems are actually secure.

### Capabilities and REST (Section 9.2)

This section relies heavily on the example of Dropbox's implementation of capability-based security. If an application were to need access to a user's dropbox file, say an image, with the OAuth2 flow, there would be no way for Dropbox to easily grant permission to a single file; OAuth2 permissions need to be generic in order to apply to all users. However, with a capabilities, it becomes easier.

In order to grant permission to files in Dropbox, it is possible to redirect the user to a page provided by Dropbox allowing them to select a file (or several files) from their account. Once they're done and they hit confirm, Dropbox then sends the capability information back to the requesting application. This scenario is ideal since it allows the user to utilize a consistent UI, the Dropbox UI, when selecting a file and, more importantly, limits the amount of access granted to only the files necessary instead of an entire account or folder. Additionally, the requesting application doesn't need to know which file it is requesting ahead of time, it just knows that it is requesting access to a file.

By nature, a session cookie is a form of *ambient authority*, or permissions granted to every request originating from a given environment. These kinds of permissions are vulnerable to *confused deputy* attacks, where a system is tricked into making a request, using their elevated privileges, on behalf of an attacker.

#### Capabilities as URIs (Section 9.2.1)

In order to create a capability string that is storable in a URL, you can utilize the strategies learned in pervious chapters. However there are some key things to keep in mind:

- Capabilities do not directly identify a user
- Capabilities are tied to a single resource and cannot be used to access other resources
- Capabilities can live longer than access tokens because their scope is limited

A very common example of a capability URI is a password reset link. You don't need to identify yourself when you follow the link since the URI itself contains all the permissions required to perform the reset.

You can easily create a capability-based URI by simply adding a token into your URL. It could be in the path, in the query parameters, or even in the [user info](https://docs.microsoft.com/en-us/dotnet/api/system.uri.userinfo?view=net-6.0) segment. When the token is an OAuth2 token, it is standardized to [use a parameter called `access_token`](https://tools.ietf.org/html/rfc6750#section-2.3).

Unfortunately there are drawbacks to using the path and query options

- The path and query segments are often logged by applications and proxies, exposing the capability to anyone with read access to the logs
- The URI may be visible through the HTTP `Referer` header or `window.referrer` in JavaScript (although using the `Referrer-Policy` header and `rel="noreferrer"` attributes on HTML links to help prevent the leakage).
- The URIs will appear in your browser history and anyone with access to your history will have access to the capabilities within

For increased security, you can place the token in one of two additional places

- the "userinfo" section `https://access_tokn@www.example.com/path/to/resource`
- the "fragment" section `https://www.example.com/path/to/resource#access_token`

Neither of these two sections are recorded in history or logged by web servers, and can be stripped out by applying the `Referrer-Policy` header and `rel="noreferrer"` attribute.

Luckily, the capabilities listed here apply when navigating a website; an API doesn't suffer from the same ailments.

- The `Referrer` header and `window.referrer` variable in JavaScript are populated by browsers
- Users don't navigate directly to API URLs, and therefore will not show the requests in their history
- API urls aren't likely to be saved or bookmarked for very long.

#### Using capability URIs in the Natter API

In order to create a capability URI, the code is simple enough to translate from Java

```c#
public Uri CreateUri(
    HttpContext context,
    string path,
    string permissions,
    TimeSpan expiryDuration
)
{
    Token token = new(
        Expiration: DateTime.Now.Add(expiryDuration),
        Username: string.Empty
    );

    token.AddAttribute("path", path);
    token.AddAttribute("perms", permissions);

    string tokenId = _tokenService.CreateToken(context, token);

    Uri result = new(context.Request.GetDisplayUrl());

    UriBuilder ub = new(context.Request.GetDisplayUrl());

    ub.Query = $"?access_token={tokenId}";

    return ub.Uri;
}
```

Instead of allowing the `Token.Username` property to be null as Madden suggests, I'm going to simply set the username field to `string.Empty`, since this will cause far fewer warnings from the nullability checking in older parts of the code.

#### HATEOAS (Section 9.2.2)

Hypertext as the Engine of Application State (HATEOAS) is a [crucial principle of REST API design](https://roy.gbiv.com/untangled/2008/rest-apis-must-be-hypertext-driven) which asserts that client interactions should be driven by links, and no prior knowledge should be required to utilize a REST API and construct URIs. All the information a client needs to utilize an API should be provided as links and form templates in the responses. All they should require is a single entrypoint-URI (a bookmark). The goal of HATEOAS is to help remove assumptions from the clients' code and allow for the API to evolve over time.

We will accomplish this by returning additional URIs in the response body after a space has been created. This is easily accomplished using an anonymous object in .NET.

```c#
return Created(
    uri,
    new 
    { 
        name = space.Name,
        uri = uri,
        messages = messagesUri
    }
);
```

However, that only provides a link to expose full permissions, which isn't ideal. So instead we should provide several links that expose a range of permissions. We can utilize anonymous functions in C# to make this a little less repetitive.

```c#
Func<string, Uri> getMessagesUri = (string permissions) => _capabilityService.CreateUri(
    HttpContext,
    path: $"spaces/{space.Id}/messages",
    permissions,
    expiry
);

return Created(
    uri,
    new 
    { 
        name = space.Name,
        uri = uri,
        messages_rwd = getMessagesUri("rwd"),
        messages_rw = getMessagesUri("rw"),
        messages_r = getMessagesUri("r")
    }
);
```

A general rule when generating a capability URL is that an action should not generate a set of permissions greater than the set required to create it (e.g. if it only took read/write permissions to perform, it should not also allow delete). You can more safely perform this task by working backwards, taking the currently allowed permissions and removing any that are not necessary (e.g. `string perms = context.Items["perms"].replace("w", "")` to remove write). I've neglected to do this in our .NET implementation because I'm a little lazy.

#### Capability URIs for browser-based clients (Section 9.2.4)

Madden's recommended way to pass the access tokens in a browser is to utilize the fragment portion of the URL to engage the [web-key](http://waterken.sourceforge.net/web-key/) pattern.

As mentioned in [Section 9.2.1](#capabilities-as-uris-Section-921), putting the token in the path or query can lead to that token being logged by browser history, web servers, or proxies. When working with a potentially unknown web server, it is therefore better to ensure that the client is the one making the API calls directly and utilizing the web server for "template" files.

The basic flow is:

1. Client requests a resource "https://example.com/resource#my_token"
2. Server returns the template found at `/resource`
3. The template then makes a new request to the API (avoiding the web server) by removing the fragment and attaching it to the query where the API expects it to be

On a personal note, this seems convoluted, and also requires that the client have direct CORS access to the API. It wouldn't really work if you were expecting your API to be extended by third parties.

#### Combining capabilities with identity (Section 9.2.5)

While capability-based controls are great, they can be further enhanced by combining them with identity-based access control.

- Add user information and claims to the token to record who is performing the action. This is ideal for short-lived links like password resets, longer duration tokens would suffer from someone being able to impersonate the user.
- Use a session to identify the user for logging/audit purposes. Because capabilities now handle permissions and access, the session can be much longer-lived, decreasing the time between user logins.

In our .NET implementation, we can switch our `ISecureTokenService` registration back to the `CookieService` since we will be following the latter school of thought from above.

#### Hardening capability URIs (Section 9.2.6)

Because capability URIs are invulnerable to CSRF attacks, you should be able to remove the anti-CSRF token created in earlier chapters. However, capabilities are vulnerable to theft; anyone with a capability URI can impersonate the user without additional securities.

That's why it's better to associate capabilities to an authenticated user, to prevent them being used by anyone else. This makes capability URIs less shareable, but improves security. 

With this change in place, the CSRF token can be removed, since the capability itself is acting as a de-facto CSRF token.

### Macaroons: Tokens with caveats (Section 9.3)

A *macaroon* is more or less an extension of a capability token, with additional "caveats" appended to restrict the capability further than normally possible. For example, you could allow a user access to read messages from a particular time frame, or from a particular user.

Caveats can be general expressions like `method = GET` or `since >= 2019-10-12T12:00:00Z` and anyone can append a caveat to an existing macaroon without calling an API endpoint.

Once a caveat has been added it cannot be removed. This is possible because macaroons use HMAC-SHA256 tags, and each time a new capability is added, the previous authentication tag is used to sign the new tag, meaning there is a chain of caveats that cannot be rewound. You cannot reverse an HMAC to recover an old tag and thus cannot revert the state of the caveats, only move forwards.

This feature of macaroons makes it important to **only use macaroons to restrict access** and never to allow access. Otherwise you would be vulnerable to users appending new permissions to their tokens.

Steps to verify the HMAC chain for a macaroon can be found in the book, including a Java example. I won't implement a .NET example at this point, since in Natter we will utilize an existing macaroon library to perform this verification.

There are two classes of caveats, first-party and third-party, which are discussed in later sections.

#### Contextual caveats (Section 9.3.1)

One feature of caveats, *contextual caveats*, allows clients to append a caveat just before use in order to restrict a token's use just before sending to an insecure or untrusted API. For instance, you could add a caveat stating that the token is valid for only the next five seconds. If the token is intercepted, the damage done is minified.

And because the original macaroon, before the contextual caveat was added, is intact, the client can continue sending requests without requiring a new macaroon.

There is no formal specification of macaroons and it isn't very widely implemented, so your mileage may vary. At this point in time, it doesn't appear that Keycloak supports macaroons [nor does it have any plans to implement support](https://issues.redhat.com/browse/KEYCLOAK-12530?jql=project%20%3D%20KEYCLOAK%20AND%20text%20~%20%22caveats%22%20OR%20text%20~%20%22macaroon%22).

#### A macaroon token store (Section 9.3.2)

In the Java-based Natter API, Madden utilizes [JMacaroons](https://github.com/nitram509/jmacaroons). For the .NET implementation we will attempt to use [Macaroons.NET](https://github.com/JornWildt/Macaroons.Net) which is a .NET port of the C [libmacaroons](https://github.com/rescrv/libmacaroons) library. It only has 23 stars on GitHub, but considering the Java port only has 100, I suppose that's not too bad.

#### First-party caveats (Section 9.3.3)

The primary type of caveats are first-party caveats, which can be verified using information the application has readily available like the current time, attributes of the request, and additional environmental features.

The most common, and seemingly only widely-used, use case for first-party caveats is to set an expiration for a macaroon, much like a JWT has, which allow users to reduce the time a token is available to shorter than the original expiration time.

First-party caveats are fast and cheap to add to a macaroon, and are therefore an effective and manageable option.

#### Third-party caveats (Section 9.3.4)

Unlike first-party caveats, a third-party caveat is verified using an external server or application. The application needing a macaroon verified would send the macaroon in to the third-party service and receive back a *discharge token* in response. These discharge tokens are bound to the original macaroon, so the application server never has to directly communicate with the third-party server, the request can be made by anyone/server at any time.

Additionally, since the discharge macaroon is still a macaroon, it can be added on to by the third-party to provide additional checks.

Third-party caveats have three components as opposed to a single string

1. Location - where to locate the third-party service
2. Secret - a unique string used to derive a new HMAC key for the signing of the discharge macaroon
3. Caveat ID - a **public** identifier for the macaroon

In order for the third-party to verify the caveat, it needs to have access to the secret and the query. There is no standard (currently) for how to perform this action, but there are a number of viable options listed in the chapter itself if you are interested in learning more. This lack of standardization means most third-party candidates are kept in-house for closed systems to communicate with one another.
