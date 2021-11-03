# Session cookie authentication

Read by **November 2, 2021**  
Presented by **Raghda**  
Implementation by **Lyn**  

## 4.1 Authentication in web browsers

In section 4.1.3, Madden introduces the concept of the *same-origin policy* (SOP) and *cross-origin resource sharing* (CORS). When making a request from a file hosted directly on the system (file://some/file/path) to the API backend (https://localhost:4567/endpoint), the browser will block the request because the API is not considered to be part of the same *origin*.

An *origin* of a URL is the **protocol**, **host**, and **port** components of that URL. In the example above, this breaks down into `{ protocol: "file", host: /some/file/path, port: -1 }` (or something similar) for the file and `{ protocol: "https", host: "localhost", port: 4567 }` for the API. Since the three properties do not match, the two endpoints are considered to **not** be the same origin and the browser automatically blocks the request.

It's important to note that CORS blocks *the response* in most cases, and not the actual request. That means that the request will still be handled by the server, the user just can't access the response. Thankfully the allowed requests are limited to using `GET`, `POST`, or `HEAD` (other requests are blocked), and also requires that there are a limited selection of headers on the request. If a Content-Type header is included, it must be either `application/x-www-form-urlencoded`, `multipart/form-data`, or `test/plain`.

This means that it is critical for sensitive endpoints to utilize more than just CORS for security. For example, a request to `POST /users/` will still create the user assuming no other protections, just the potentially malicious attacker won't know if it was successful or not (and may not care). Madden says we'll learn more about CORS in chapter 5 and how to mitigate issues like this.

In order to avoid CORS issues in scenarios like these, the web server itself must deliver the files. In ASP.NET this is accomplished through a number of methods, but most commonly by putting static assets in a `wwwroot` folder at the root of the project and adding the `UseStaticContent` middleware to the pipeline. This means that the files will be served from the same origin and the requests will now be valid.

By default, when a website returns the `WWW-Authenticate` header with the basic authentication scheme, most browsers will then show a standard login popup with a prompt for username and password. Once this has been filled out, the "[browser will] remember HTTP basic credentials and automatically send them on subsequent requests to the same URL path." This isn't ideal for multiple reasons

- Every request will contain the user's username and password, increasing the possibility of those details being intercepted and decoded
- Verification of the credentials will happen on every request, meaning the purposefully slow hashing algorithm will slow down every endpoint
- The default login forms are very simple and don't provide a very good UX
- There is no obvious way to "logout" with this method
- Browsers will only include the credentials for "siblings" of the original URL, so any parent paths will not include the authentication

To mitigate this, we will perform a number of alterations to the site, including implementing a cookie-based session. But it's also worth noting that [HTTP Bearer authentication from OAuth2](https://tools.ietf.org/html/rfc6750) is gaining popularity.

## 4.2 Token-based authentication

For token-based authentication, the user only has to send in their credentials once. The web server(s) then acknowledges the credentials and returns a randomly generated string to server as a session token. The user can then use this token in place of their credentials to verify their identity with every request. A good analogy for this are modern hotel room keys, which are generated when you check in, allow you access to everything you're allowed to have access to (and nothing more), and automatically expire when your stay comes to an end (expires).

This makes it more practical to have a dedicated authorization server/microservice that is only concerned with logging in users which other microservices can then interact with (or interact with its datastore directly) to determine the validity of any tokens.

For the NatterAPI, we'll be handling this token store with an in-memory database (in the author's implementation this is a Spark web server's built-in feature).

# 4.3 Session cookies

To begin, we'll implement our token-based authentication with the most common methodology, cookie-based authentication.

As per the standard, once a user logs in from the login endpoint, the server will return a `Set-Cookie` header on the response. The web browser will then store this cookie and send it in on every subsequent request automatically.

Initially, the NatterAPI utilizes Sparks' `request.session(true)` to check if a session exists and create it if it does not. However, this introduces a vulnerability called _session fixation_.

> **Vulnerability**: Session fixation
> 
> When an attacker can hijack a user's account by using an existing session. The attacker can create a session by logging in to the API server, then send the user a link that includes that session ID in the URL. During authentication, the server finds the session successfully (because the attacker created it previously), and the session information is then updated with the victim's data. The hacker then can continue using their session token, which is now the token for the victim, allowing them to access all the victim's secure assets.

To avoid this issue, simply re-create the session on every login request. In Spark this is accomplished by utilizing `var session = request.session(false)` to attempt to get the session, then check if that session was `null`. If it was not `null`, then simply destroy or invalidate the session via `session.invalidate()`.

Once using cookies, there are a handful of attributes that can be attached to the cookie to increase the security

- `Secure` - cookie can only be sent over HTTPS. The browser will refuse to send the cookie if the connection is not secure.
- `HttpOnly` - cookie **cannot** be accessed from JavaScript. The browser will not return the cookie in the list of cookies when requested from JavaScript.
- `SameSite` - only requests from the same origin are included in the request
  - Enabling this can decrease attack surface area by preventing request from unknown sites
- `Domain` - without the `Domain` attribute, the cookie will only be sent on requests matching the exact host. The `Domain` attribute will add the cookie to requests on all sub-domains as well (e.g example.com will allow the cookie to be sent for api.example.com)
  - Don't use this one unless you **really** need it to be shared with a sub-domain.
  - When included, only **one** sub-domain needs to be compromised to effect the entire domain
- `Path` - similar to the way browsers handle basic authorization, the `Path` attribute defines which path(s) the cookie is included on. When included, the cookie will only be included on the original path and sub-paths of that path. Given `Path=/spaces`, it will be sent to `GET /spaces` and `GET /spaces/1` but not `GET /users` because `/users` is a different path.
  - This one provides minimal benefits as it is easily defeated by using an iFrame.
- `Expires` and `Max-Age` - the expiration date for the cookie
  - `Max-Age` is newer and preferred, but `Expires` is used for backwards compatibility with Internet Explorer
  - Once a cookie is past its max-age, it is removed from the browser

A token with a `Max-Age` or `Expires` attribute is considered a **persistent** token and will be kept around until the the expiration date is reached. Without those attributes, the token will be automatically deleted once the user closes the browser. For this reason, you should avoid using persistent tokens for your session tokens, especially on shared devices. Many websites only include this attribute if the user clicks the "remember me on this computer" button.

"**You should always set cookies with the most restrictive attributes that you can get away with**". At minimum, a security cookie should include the `Secure` and `HttpOnly` attributes.

> **Vulnerability**: Sub-domain hijacking or sub-domain takeover
> 
> Generally caused when a shared service, like GitHub Pages, creates a sub-domain of the main website that is deleted once it is no longer needed. The service deletes the application but can often forget to remove the DNS records. An attacker can then use those DNS records and re-register the site on the shared web host, allowing them to serve their content from the compromised sub-domain.

For increased security, [all browsers except for Internet Explorer](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie#browser_compatibility) support special prefixes for cookie names. The "`__Secure-`" prefix mandates that the cookie is set with the `Secure` attribute and are served via HTTPS. Meanwhile, the "`__Host-`" prefix increases the security from `__Secure-` by additionally requiring that the cookie has no `Domain` attribute, which helps prevent sub-domain hijack attacks. An example of utilizing these prefixes could look like "`__Host-NatterSession`".

## 4.4 Preventing Cross-Site Request Forgery attacks

The main drawback of utilizing cookies is that the browser will include those cookies on *any* request made to the URL they originated from. This means that if an attacker sends a request to your secure website from their own domain, the cookie will still be included. While the *Same-Origin Policy* ensures that the malicious script cannot read the response, it does nothing to stop the original request itself, allowing the hacker to do malicious things as long as they don't require the response. This kind of attack is called a *Cross-site request forgery* attack.

> **Vulnerability:** Cross-site request forgery
> 
> Also known as CSRF or XSRF, this attack occurs when an attacker sends a request to an API from another origin, including the secure cookies with the request. If no extra checks are performed, then the request is fulfilled and the response is sent back to the browser (which then hides the request from the sender via CORS).

For our JSON-based API example, setting the Content-Type to `application/json` makes it more difficult to carry out a CSRF attack, since CORS will not allow requests from a different origin that do not match one of the three allowed Content-Types [see 4.1 Authentication in web browsers](#4.1-authentication-in-web-browsers) above. However, attackers are able to get around this basic protection by utilizing vulnerabilities in Adobe Flash.

The most basic protection you can make against CSRF attacks is to "**never perform actions that alter the state on the server or have other real-world effects in response to a `GET` request.**"

Another protection you can use is the `SameSite` attribute for cookies, which ensures that cookies are only included on requests to the the same *registerable domain* that set the cookie.

> **Definition:** A *registerable domain* is simply the host of the URL, not including protocol and port like an origin. For examples, "https://example.com" and "http://api.example.com" are the same while "https://api.example.com" and "https://api.example.net" are different.
> 
> Registerable domains also include some sub-domains, known as *effective top-level domains* (eTLDs), but these are not well-defined with a rule and are instead included on Mozilla's *public suffix* list found at <https://publicsuffix.org>.

`SameSite` allows two different values, `lax` and `strict`. In `strict` mode the cookie will not be included on any cross-site requests. This is great from security's perspective, but this can also cause issues when clicking a link from one site to another; for example clicking a link to go to the NatterAPI from another site would mean that the user's current session isn't utilized. To somewhat ease this issues, the `lax` option includes the cookie on requests when a user directly clicks on a link, blocking all other requests. Obviously `strict` is preferred, but it may not be easily manageable. However, if you're using an SPA, then this isn't necessarily an issue because the entire site is served from a single origin. [The OWASP foundation has a potentially more helpful description of this attribute here](https://owasp.org/www-community/SameSite).

It's worth noting that in most modern browsers, `SameSite=lax` is the default value.

Even though `SameSite` helps, it doesn't prevent the sub-domain attacks; the weakest sub-domain is still a threat to your entire site. You can improve your security even further by implementing _hashed-based double-submit cookies_.

### Hash-based double-submit cookies

To increase security further, you can ensure that a request must verify their session token *twice* in order to be considered authenticated. This is done by requiring that not only the cookie be sent in, but also the session ID via the `X-CSRF-Token` header. 

When a login request is sent to the API and is successfully completed, the API should include the `CSRF-Token` in the response. The consumer should then store that token in a new cookie that is accessible from JavaScript.

Now, whenever a new request is made, the JavaScript should get the CSRF cookie from the jar and include it in the request via the `X-CSRF-Token`.

```javascript
// response from login
const response = await fetch("/login");

if (response.ok) {
    const json = await response.json();

    // store the CSRF token in a new cookie
    document.cookie = `csrfToken=${json.token};Secure;SameSite=strict`;
}

// new request

// the `getToken` function is a custom function, but a library can be used
const csrfToken = getCookie("csrfToken"); 

fetch("/doStuff", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken // include the token
    }
});
```

However, using the session ID as the CSRF Token is only partially secure, as the header can still be overwritten. A better implementation of the CSRF Token is to utilize a *cryptographically bound* version (or hash) of the session ID.

By hashing the token before returning it to the user, you make it much harder for the original session ID to be guess and the token to be faked.

One final note about hashing, whenever you perform a hash comparison, you should always do it in constant time, in C# this can be accomplished with the [CryptographicOperations.FixedTimeEquals(byte, byte)](https://docs.microsoft.com/en-us/dotnet/api/system.security.cryptography.cryptographicoperations.fixedtimeequals?view=net-5.0) method. This is necessary to prevent [timing-based attacks](https://owasp.org/www-pdf-archive/2018-02-05-AhmadAshraff.pdf); when an attacker sends in a hashed string and the comparison is performed, they can inspect minute differences in the time a comparison took to determine whether or not they have the correct characters. If a comparison of two hashes ends early, because a difference is found, then the attacker can use the timing differences to determine how many of their characters are correct. Continuing this pattern, they can determine the entire hashed value with minimal guesses.

Graham had a useful analogy for this: take for instance a combination lock. Each time you get a component of the combination correct, the handle will give just a little further. You can try each position until you get the number that gives the most, then move on to the next position. Eventually you will have cracked the combination without guessing every combination in existence.
