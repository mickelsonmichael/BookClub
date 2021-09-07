# Designing a network-efficient API

## Summary & Notes

### Identify Problems

- Users complain that it is too slow, drains their battery, and uses too much of their data allowance.
- API providers complain about Cloud providers bills.

### Reasons

- The number and frequency of network calls and the volume of data exchanged.

### Explanation

In the example scenario network bandwidth is 160KB/s, latency (delay in communication over a network) is always 300ms, the process time is 20ms, and the response time is 200ms.
This size of API call is fine until you need to make multiple API calls at once. At 25 API calls, you get upwards of a five second delay.

### Network Efficiency Factors ARe

1. Speed
2. Data volume
3. Number of calls

What we need is to find a balance between the need for efficiency and an ideal design

### 10.2 Network communication optimization/enhancing

For HTTP-based APIs:

- Activating compression and persistent connections can reduce data volume and latency
- Enabling caching (letting consumers know if they can ..)

### 10.2.1 Activating compression and persistent connections

If the Banking API is hosted on an on-premises infrastructure, less data means less risk of network congenstion because the overall bandwidth used is lower and network connections will be open for less time. For loud infrastrucutres, this simply means smaller bills.

Once persistent connections are enabled on the API server

- the user case that required 25 calls distributed in seven steps taking a total of 4.2s can be reduced to 2.4s by removing 6 x 300ms of latency
- only the first API call will have to suffer the connection latency, the subsequent calls are made using the same connection. The connection stays open for a given number of calls or a given time, determined by the server's configuration

Tag/Headers used

The `Cache-Control` header's value is `max-age=300`, which means that this response can be cached for 300s (5 minutes).
So, if the application needs to show the account's data again in the next five minutes, it will use the cached response instead of making a call to the API.

"Give me account A1's data only if it has been modified." This is done using the `2557dff` ETag value returned by the first call.
An `If-None-Match: "2557dff"` header is sent along with the second request.

> If the data has not been modified, the server returns a `304 Not MOdified` response without any data and, therefore, avoids loading data unnecessarily.

Bottom line:

> An API can enable the caching of its responses and propose conditional requests that greatly optimize network communications by reducing the number of calls and the volume of data returned. Caching also guarantees a certain freshness and accuracy of the data.

### 10.2.3 Choosing cache protocl policies

- How long data can be cached can depend on how often it is updated
- The appropriate cache duration should be determined not only by how the Banking system works but also by how people actually use their bank accoutns
- Presenting an innacurate balance, even through a third-party application, can cuase some problems from a legal or security perspective. Therefore, the Banking API's documentation might state that consumers must use fresh data. As with real-time balance information, in such a scenario, the API would provide a `Cache-Control ` header with a value of `0` but can still make use of condition request.

### 10.3 Ensuring network communication efficiency at the design level

Some strategies we can use to optimize the number of calls and the volume of data exchanged when the API is used by the consumer

### 10.3.1 Enabling filtering

Proposing filtering options makes an API more usable and more efficient by allowing consumers to request only the data they actually need - and this is what we desperately need here. Every byte saved improves communication eficiency in a hostile network context.

So, consider the nature of the data and context and use.

### 10.3.2 Choosing relevant data for list representations

Choosing a relevant representation including the most representative and useful properties of a resource is the bset way not only to create usable API but also to avoid many API calls after getting the list's data.

There are cases when returning a complete representation is more efficient.

### 10.3.3 Aggregating data

Instead of several short calls, make one longer one. The response time is cut by almost 709%!

Example: Retrieving all account data except the transactions in one call.

### 10.3.4 Proposing different representations

Capability of providing different representations of a resource, three different levels of representations of our resources

1. Summarzied
2. Complete
3. Extended

Each one correponds to an `application/vnd.bankingapi.{representation}+json` media type (i.e. `application/vnd.bankingapi.extended+json`).

The summarized representation provides a subset of the complete representation's data. It's the one we are used to getting in lists using the list owners' goal (`Get /owners`, for example).

Finally, the extended representation ...

### 10.3.5 Enabling expansion

Utilize a query parameter to define additional fields that can be included.

### 10.3.6 Enabling querying

- Consumers can easily select exactly the data they want and make multiple queries in a single call.
- Enabling data querying might be appropriate in some scenarios, but not all. It can reduce the volume of data transferred and the number of API calls, but at the possible expense of caching possibilities.

### 10.3.7 Providing more relevant data and goals

- Providing relevant data means not providing all the available data. Indeed, focusing on the consumer's perspective can help to limit data volumes
- Adding more goals providing different access to the same resources or more direct access can also improve usability and efficiency in different context
- Adding dashboard view as a different view

### 10.3.8 Creating different API layers

Optimizing an API design in order to provide effficient communication must not be done at the expense of usability and resuability. Trying to please all consumers by making specific modifications here and there or adding multiple highly-specific goals will probably lead to a complex API that will not be reusable.

#### Type of API exists

- **Backend for frontend** - consumers have specific needs they should build their own APIs on top of the provider's
- **Experience APIs** - specialized APIs creating a new API layer in their systems. (Rest, GraphQL, or whatever) their design is optimized for a specific context of use from a functional or technical (network, usually) perspective.
- **Original/Not specialized APIs** - these are APIs whose design is consumer-oriented but that are not really confied to a specific context of use
- **System APIs** - providing access to core systems. If you remember the microwave oven.

## Discussions

## Exercises

1. As a group, determine the "ideal" cache times for a hypothetical online book store called "Brooke's Books", which provides a `GET /books` endpoint that lists all the books available.
The endpoint will aim to always provide only the latest edition by the latest publisher.

| Property | Cache Limit |
| -------- | ----------- |
| ISBN | inifinte |
| Author | infinite |
| Title | infinite |
| Rating out of 10 | 24 hours |
| Price | 0 |
| Edition | infinite |
| # In Stock | 0 |

2. Given the decisions above, what would be a good cache duration for the `GET /books` endpoint?

`Cache-Control: 0`

But if some are zero while others are infinite, should they be split into multiple endpoints with different caches?

3. If the book library is rather large, but you want consumers to be able to filter that library, which of the following remediations might be most appropriate?
  - [x] Count-based pagination - see footnote
  - [ ] Cursor-based pagination
  - [ ] Property filtering
  - [ ] Custom filtering/queries

For the pagination options, they would only be usable if the list of books has a particular order. If they are ordered, then cursor-based may be overkill, since the list of books don't change very often. If you chose to put new books at the end of the list (since we're picking the ordering) then you don't have to worry about the issues with count-based.

Is property filtering really ideal? Or should there be multiple endpoints for this API. It increases the complexity of the API and makes query strings more complex.

4. If you wanted to implement the `etag` header, what might an appropriate hash be?

### Caching

#### ASP.NET Core

[Response caching in ASP.NET Core is accomplished several ways](https://docs.microsoft.com/en-us/aspnet/core/performance/caching/response?view=aspnetcore-5.0) but can be automated using the `ResponseCacheAttribute` class and the [ResponseCaching Middleware](https://docs.microsoft.com/en-us/aspnet/core/performance/caching/middleware?view=aspnetcore-5.0).

```c#
// Startup.cs
public void ConfigureServices(IServiceCollection services)
{
  services.AddResponseCaching();
}

public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
{
  app.UseResponseCaching();
}

// Controller.cs
[ResponseCache(Duration = 30)]
public class SomeController : Controller
{
  //...
}
```
There is however a warning flag on the documentation for this attribute that should be noted.

> Disable caching for content that contains information for authenticated clients. Caching should only be enabled for content that doesn't change based on a user's identity or whether a user is signed in.

However, the middleware should prevent this automatically if it detects the `Authorization` header on the response.

#### NodeJS

You can utilize [ExpressJS](http://expressjs.com/en/api.html#express.static) and their built-in caching feature for static files like below.
Simply pass the `static` method a directory path to your static assets, which will then be cached for a maximum of the value provided in `maxAge`.

```js
app.use(
  express.static("static/file/dir", {
    maxAge: "5m"
  })
)
```

If you want to cache the results of requests with inputs, not just static files, then [ExpressJS recommends that you utilize a "caching server like Varnish or Nginx"](https://expressjs.com/en/advanced/best-practice-performance.html#cache-request-results).
They have provided a [helpful link to serversforhackers.com walking through nginx caching](https://serversforhackers.com/c/nginx-caching).

### ETags

#### ASP.NET Core

There is no out-of-the-box support the ETag header using ASP.NET Core, but there are libraries like [CacheCow](https://github.com/aliostad/CacheCow) which can provide a more automated caching experience.

#### NodeJS

For ExpressJS, you can utilize the `app.enable('etag')` call to enable etags automatically.

```js
app.enable("etag");

app.set("etag", "strong"); // default, uses "strong" etags
app.set("etag", "weak");
app.set("etag", function(chunk, encoding) { return etag; }); // custom tagging function
```

The `strong` and `weak` keywords are related to the default hash function.
You can see [this discussion thread](https://github.com/expressjs/express/issues/2129#issuecomment-43965315) for more information on the strengths.

Alternatively, many people seem to recommend the [fresh](https://www.npmjs.org/package/fresh) npm package.

### Filtering

An example of filtering only the properties required can be found on the [`Get issue` endpoint from JIRA](https://docs.atlassian.com/jira-software/REST/7.3.1/#agile/1.0/issue-getIssue), allowing the user to request only a certain set of paramters from the request.
The consumer can filter these results even further by defining the exact properties returned for each issue by using the `fields` paramter.
This turns what could be a very large dataset, including history tracking for every issue, into a much more manageable size.
Finally, they allow you to filter the issues using a custom "jql" query language, which works similarly to SQL to provide only the issues matching certain requirements.

Here is an example you could use to limit all the issue information to only get the history of changes in the last 24 hours for a particular project.

```powershell
# Get the issues for "Project1" updated in the last 24 hours.
$jql = "project=`"Project1`" and updatedDate > `"-24h`";

# Only return the changelog data
$expand = "changelog";

# Only return the fields needed to parse the change events
$fields = "creator,status,description,updated,summary"

$response = Invoke-WebRequest "${RootUrl}search?jql=${jql}&expand=${expand}&&fields=${fields}
```

For identity-based pagination, you can see an example of this pretty easily when [browsing Reddit's "old" layout](https://old.reddit.com/user/spez?count=25&after=t1_g4lb2e2).
When paging through a user's posts, in this example the Reddit Administrator `u/spez`, you can see the URL query being updated with the current count and the index to continue from, like "?count=25&after=t1_g4lb2e2".
In this instance, count is the number of posts to return, while the `after` key likely indicates the ID of the index post.

Interacting with this directly will yield you the same results each time, because the Reddit UI is wrapping the [Reddit API, which hasa method you can use to preview the actual functionality](https://www.reddit.com/dev/api/#GET_user_{username}_overview).
