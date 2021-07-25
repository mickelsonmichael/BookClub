# What is API Design?

## 1.1 What is an API?

- This book is about a remote API, not a hardware or library
- For this book, "an API is a web interface for software"
  - "a point where two systems, subjects, organizations, and so forth meet and interact"
  - An API is "only and interface exposed by some software" and is "an abstraction of the underlying implementation"
- There are _consumers_ and _providers_
  - _Consumer_ is the consumer of the API
  - _Provider_ is the backend
- This book considers only HTTP APIs as Web APIs
- An API can rely on multiple other smaller APIs, both private and public
- APIs should be developed with multiple consumers in mind; don't just code for Web or just for Mobile
- An API should be able to run independently as long as there is a network connection
  - This allows different services to be installed on more resource-intense servers
- Public vs Private APIs
  - "Public APIs are proposed _as a service_ or _as a product_ by others"
  - "A private API is one you build for yourself: only applications created by you or people inside your [organization] use it." You are both the _provider_ and the _consumer_.
  - Public vs private "...is not a matter of _how_ an API is exposed, but _to whom_." A private API can still be exposed to the public internet, but not be accessible to anyone outside the organization.
  - An _almost public_ or _partner_ API is a private API where some of the endpoint are exposed to customers or select partners

## 1.2 Why API design matters

- APIs are used by _people_, so if they are poorly designed, it results in frustration and unhappiness
- Even _private_ APIs may be used by other teams within the company, so they too must be well thought out
- _developer experience (DX)_: "the experience developers have when they use [software]" (in this case an API)
  - registration, documentation, support, and most importantly usage (the design)
- An API should be designed so as to hide the implementation details
  - The user(s) should not know _how_ the implementation is handled, only what the result should be
  - User(s) shouldn't even need to know what language the implementation is written in
- "...hiding the implementation isn't enough. It's important, but it's not what people using APIs really seek"
- "A poorly designed product can be misued, underused, or not used at all."
- Often times, teams are forced to use terrible private APIs or even public APIs
  - Increases time, capital, and effort required to interface
- A poorly designed API can lead to unhappiness from the users, leading to less adoption
  - Could also result in more calls to the support desk
- Even worse, a poor implementation could expose security vulnerabilities

## 1.3 The elements of API design

- **API styles** include [RPC](https://en.wikipedia.org/wiki/Remote_procedure_call), [SOAP](https://en.wikipedia.org/wiki/SOAP), [REST](https://en.wikipedia.org/wiki/Representational_state_transfer), [gRPC](https://en.wikipedia.org/wiki/GRPC), and [GraphQL](https://graphql.org/)
- All the API styles share a core set of fundamentals that you should learn
- Changes to a design should be done with extreme caution
  - New versions that are too dissimilar to previous versions may not appeal to users
