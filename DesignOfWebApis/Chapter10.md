# Designing a network-efficient API

## Discussions

## Exercises

1. As a group, determine the "ideal" cache times for a hypothetical online book store called "Brooke's Books", which provides a `GET /books` endpoint that lists all the books available.
The endpoint will aim to always provide only the latest edition by the latest publisher.

| Property | Cache Limit |
| -------- | ----------- |
| ISBN | |
| Author | |
| Title | |
| Rating out of 10 | |
| Price | |
| Edition | |
| # In Stock | |

2. Given the decisions above, what would be a good cache duration for the `GET /books` endpoint?

3. If the book library is rather large, but you want consumers to be able to filter that library, which of the following remediations might be most appropriate?
  - [ ] Count-based pagination
  - [ ] Cursor-based pagination
  - [ ] Property filtering
  - [ ] Custom filtering/queries

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
