# Capability-based Security and Macaroons

- (Michael's Notes)[#michael]

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
