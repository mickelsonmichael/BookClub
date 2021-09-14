# Designing an API in context

Other factors along with network efficiency may influence API design.
- Habits and limitations (E.g. Legacy systems that only work with XML)
- Standards (E.g. ISO 20022 for banking)
- Provider perspective (E.g. Not being abl to wrap an asynchronous or multi-step process in a synchronous one-step goal)

## 11.1 Adapting communication to the goals and nature of the data

The processing of a goal may include long (or even human-interaction) steps, which makes blocking for a single request unfeasible.

In this case, return a status code `204 Accepted` from the initial request along with a `self` URL. Then allow subsequent GET requests to the URL to reveal the current status.

Be sure to set appropriate cache durations on the response to the GET requests, as this pattern will lead to repeated requests.

The above polling can be avoiding using a "reverse API" (also known as a "webhook").

The interface to the webhook (i.e. the content of the POST requests) is defined by the provider. The service is implemented by the consumer.

Simple webhook interfaces are easier to implement by consumers.

For security, the information contained in the webhook should be very limited. To get detailed information, the consumer is required to use the normal API.

The WebSub W3C standard provides guidelines on how an API provider can expose information about their webhook interface and to allow consumers to register for events securely.

For continually changing data, both polling and webhooks may be poor choices compared with Server-Sent Events (SSE). In SSE, the response has an content type of `text/event-stream` and never terminates (until the connection is closed).

When using SSE, providing a "complete" data set is better, unlike the recommendation for webhooks to provide minimal event data. This prevents the consumer needing to make additional requests.

In the SSE specification:
- Events are written to the stream with a prefix of `data:`, followed by the content of the event. As the content-type is text, the content can be plain text, or string-encoded JSON, XML or anything else that can be string-encoded.
- Comments start with a prefix of `:` followed by the comment.
- Events can optionally be prefixed by a line that specifies the ID of the event. E.g. `id: 123`.
- Events can optionally be prefixed by a line that specifies the type of the event. E.g. `event: it happened again`.
- A minimum retry interval in case of a lost connection can be specified by writing a line prefixed by `retry:` followed by a number of milliseconds.

As SSE is uni-directional, the stream can be compressed using standard HTTP compression. It does not rely on any additional infrastructure, other than to ensure the servers can handle the neccessary number of long-lived connections.

For bi-directional streaming, consider WebSockets. WebSockets create a TCP-level connection, which may not play nicely with proxy servers.

Another factor in API design may be the need to efficiently update multiple resources to achieve a single goal. For example, marking many items as "read". In this case, a single PATCH request that contains a list of items to update may be preferable than making one PATCH request for each item. The OpenAPI schema allows arrays to be defined with a `maxItems` property to limit request sizes and/or processing time.

When processing multiple items, if some items were successfully processed and others were not, use the response code `HTTP 207 (Multi-status)`. If however it does not make sense to partially process a list, fall back to a traditional HTTP `4xx` response.

## 11.2 Observing the full context

If the consumer of the API exists in a context with standards, they may prefer that APIs conform to those standards to promote ease of consumption in their COTS products. Talk to your customers to learn about their needs early on.

## 11.3 Choosing an API style according to the context

Don't assume the only tool you have is the best tool for the job.

- REST APIs provide a framework for common understanding.
- GraphQL is well suited for multiple provate clients that need to query unknown subsets of data, where caching is not required.
- gRPC is good for fast duplexed communication.

Also consider alternatives such as Message Queues (RabbitMQ, MQTP, etc) or streaming (SSE, WebSockets, Kafka, etc).
