# Microservices Patterns

- [Microservices Patterns](#microservices-patterns)
  - [Chapter 1 - Escaping monolithic hell](#chapter-1---escaping-monolithic-hell)
    - [Articles and Resources](#articles-and-resources)
    - [Books](#books)
  - [Chapter 2 - Decomposition Strategies](#chapter-2---decomposition-strategies)
    - [Articles and Resources](#articles-and-resources-1)
    - [Books](#books-1)
  - [Chapter 3 - Interprocess communication in a microservice architecture](#chapter-3---interprocess-communication-in-a-microservice-architecture)
      - [Synchronous messaging](#synchronous-messaging)
      - [Asynchronous messaging](#asynchronous-messaging)
    - [Articles and Resources](#articles-and-resources-2)
    - [Books](#books-2)
  - [Chapter 4 - Managing transactions with sagas](#chapter-4---managing-transactions-with-sagas)
    - [Types of Countermeasures](#types-of-countermeasures)
      - [Semantic lock](#semantic-lock)
      - [Commutative updates](#commutative-updates)
      - [Pessimistic view](#pessimistic-view)
      - [Reread value](#reread-value)
      - [Version file](#version-file)
      - [By value](#by-value)
    - [Articles and Resources](#articles-and-resources-3)
  - [Chapter 5 - Designing business logic in a microservice architecture](#chapter-5---designing-business-logic-in-a-microservice-architecture)
    - [Notes](#notes)
      - [Transaction script pattern](#transaction-script-pattern)
      - [Domain model pattern](#domain-model-pattern)
      - [Domain-Driven Design](#domain-driven-design)
      - [Aggregate Rules](#aggregate-rules)
      - [Domain events](#domain-events)
    - [Articles and Resources](#articles-and-resources-4)
  - [Chapter 6 - Developing business logic with event sourcing](#chapter-6---developing-business-logic-with-event-sourcing)
    - [Articles and Resources](#articles-and-resources-5)
  - [Chapter 7](#chapter-7)
  - [Chapter 8](#chapter-8)
  - [Chapter 9 - Testing microservices (part 1)](#chapter-9---testing-microservices-part-1)
    - [Consumer-driven contract testing](#consumer-driven-contract-testing)
    - [Articles and Resources](#articles-and-resources-6)

## Chapter 1 - Escaping monolithic hell

- "[Monoliths are] too large for any developer to understand [after a certain point]" (page 4)

> Potentially a good test of whether your microservices have a dependency on one another would be to ask "Can I change this microservice without knowing about the existence of the others that depend on it"

- Patterns have three major components
  1. **Forces**: issues that must be addressed, and may be conflicting
  2. **Resulting context**: the consequences of applying the patterns (also has three parts)
     1. _Benefits_: benefits to the pattern and _forces_ that have been resolved
     2. _Drawbacks_: drawbacks of the pattern and _forces_ that have not been resolved
     3. _Issues_: new problems or _forces_ that have been introduced
  3. **Related patterns**: relationships between patterns
     1. _Predecessor_: the _pattern_ that motivated the need for the current _pattern_
     2. _Successor_: a _pattern_ that solves an _issue_ in this pattern's _resulting context_
     3. _Alternative_: a _pattern_ that can be chosen instead of the current _pattern_
     4. _Generalization_: a _pattern_ that is a general solution to problems
     5. _Specialization_: a _pattern_ that is a specialized form of another _pattern_

> What is the difference between a _successor_ and a _generalization_? They seem very similar. Point (4) seems to describe the qualities of a pattern rather than relationships

- The _generalization_ and _specialization_ descriptions are used to say that a pattern is a _specialization_ or _generalization_ of another, so it is still a relationship. A pattern **A** can be a _specialization_ of pattern **B**, which would mean that pattern **B** is a _generalization_ of pattern **A**.
- The Microservices pattern introduces three layers of problem areas:
  1. _Infrastructure patterns_: problems outside of development
  2. _Application infrastructure_: problems partially related to development
  3. _Application patterns_: problems fully related to development
- Two _alternative_ patterns for decomposing services
  1. Decompose by business capability: organize services around business capabilities
  2. Decompose by subdomain: organize services around DDD subdomains

> **Alert**: Domain-Driven Design has been mentioned

- Five groups of interprocess communication patterns (IPC)
  1. _Communication style_: the type of IPC mechanism
  2. _Discover_: how clients figure out how to communicate
  3. _Reliability_: ensuring communication between services
  4. _Transactional messaging_: integrating messages and events with database transactions
  5. _External API_: how do clients communicate with the services
- Six patterns for creating observable microservices
  1. _Health Check API_: endpoint for exposing the status of a microservice
  2. _Log aggregation_: combining logs to a centralized store
  3. _Distributed tracing_: trace requests between services
  4. _Exception tracking_: track and filter exceptions
  5. _Application metrics_: counters, gauges, etc.
  6. _Audit logging_: logging user actions
- Three testing patterns
  1. _Consumer-driven contract test_: services meet client expectations
  2. _Consumer-side contract test_: clients can communicate with a service
  3. _Service component test_: test a service in isolation

### Articles and Resources

- [_The Ebay Architecture: Striking a Balance Between Site stability, Feature Velocity, Performance, and Cost_](https://www.slideshare.net/RandyShoup/the-ebay-architecture-striking-a-balance-between-site-stability-feature-velocity-performance-and-cost)
- [_Suck/Rock Dichotomy_ by Neal Ford](http://nealford.com/memeagora/2009/08/05/suck-rock-dichotomy.html)
- [_Gartner hype cycle_](https://en.wikipedia.org/wiki/Hype_cycle)
- [_Conway's law_](https://en.wikipedia.org/wiki/Conway%27s_law)
- [_Inverse Conway manuever_](https://www.thoughtworks.com/radar/techniques/inverse-conway-maneuver)
- [_Continuous Delivery_](https://continuousdelivery.com)
- [_Puppet: State of DevOps Report_](https://puppet.com/resources/whitepaper/state-of-develops-report) ([2021 Report](https://puppet.com/resources/report/2021-state-of-devops-report))
- [_Velocity 2011: Jon Jenkins, "Velocity Culture"_](https://www.youtube.com/watch?v=dxk8b9rSKOo)
- [_How We Build Code at Netflix_](https://netflixtechblog.com/how-we-build-code-at-netflix-c5d9bd727f15)

### Books

- _The Art of Scalability_ by Michael Fisher
- _The Mythical Man-Month_ by Fred Brooks
- _The Righteous Mind: Why Good People are Divided by Politics and Religion_ by Johnathan Haidt
- _A Pattern Language: Towns, Buildings, Construction_ by Christopher Alexander
- _Design Patterns: Elements of Reusable Object-Oriented Software_ by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides
- [_Managing Transitions_ by William Bridges, Susan Bridges](https://wmbridges.com/books)

## Chapter 2 - Decomposition Strategies

> The **software architecture** of a computing system is the set of structures needed to reason about the system, which comprise software elements, relations among them, and properties of both.

- Functional requirements: what the application must do (user stories)
  - Can be met with any architecture without a noticeable difference
- Quality of service requirements: runtime qualities, the "-ilities"
  - Scalability, testability, maintainability, deployability, etc.
  - Heavily impacted by the selected architecture

> An architectural style, then, defines a family of such systems in terms of a pattern of structural organization. More specifically, an architectural style determines the vocabulary of components and connectors that can be used in instances of that style, together with a set of constraints on how they can be combined.

- Architectural styles
  - Layered: three layers with different responsibilities
    1. Presentation layer: User interfaces and external APIs
    2. Business logic layer
    3. Persistence layer: databases and interacting code
  - Hexagonal: business logic at the center with multiple inbound adapters and outbound adapters
    - External consumers invoke the inbound adapters to perform tasks
    - Business layer invokes outbound adapters to communicate with repositories or other APIs
    - Business layer has one or more _ports_ which can be inbound or outbound and interface with adapters
  - Monolithic: application is a single executable/component
  - Microservice: application is composed of multiple executables/components that communicate through protocols
- Hexagonal and microservice styles are not mutually exclusive, and often microservices and implement the hexagonal architecture internally

> A **service** is a standalone, independently deployable software component that implements some useful functionality

- Two types of operations
  - Commands: actions and updates
  - Queries: retrieve data
- Services can also publish events

> A service’s API encapsulates its internal implementation. Unlike in a monolith, a developer can’t write code that bypasses its API. As a result, the microservice architecture enforces the application’s modularity.

- A typical service in a microservice architecture follows the hexagonal structure
- A service has two requirements:
  1. Have an API
  2. Be independently deployable

> You must treat a service’s persistent data like the fields of a class and keep them private.

- Not sharing databases has benefits but also drawbacks
  - Not sharing means locks are easier and data is simpler
  - Not sharing also means that keeping data in sync between services is more difficult
- Shared libraries should be reserved for functionality that isn't likely to change

> For example, in a typical application it makes no sense for every service to implement a generic `Money` class

- Microservice is a bit of a misnomer, in that the actual size of the service isn't important
  - The service should be manageable by one team and be independent
- Beware the _distributed monolith_

> Like much of software development, defining an architecture is more art than science.

- When designing the API of a service, use generic _system operations_ rather than specific protocols like REST to help keep the design flexible
  - List them all out for the entire system, then divide them between services
- Decompose services by _business concepts_ rather than technical
  - Seems to be contradictory to the previous book the club read on microservices, in a way. Need more information
- Obstacles to decomposition:
  1. Network latency
  2. Reduced availability
  3. Data consistency across services
  4. God classes
- Two steps to defining system operations:
  1. Create a high-level domain model
  2. Identify system operations in terms of domain model

> The domain model is derived primarily from the nouns of the user stories, and the system operations are derived mostly from the verbs

- Each service will have its own domain model, but there still remains a high level model
- Commands should have several characteristics:
  - Parameters
  - Return value
  - Preconditions
  - Post-conditions
- Decompose by business capability:
  - Business capability: something the business does to generate value and define what an organization does
  - Business capabilities are stable and don't really change, while the methods they are accomplished by do change
  - Often focused on a particular object of the domain
  - Results in a stable architecture because the underlying drivers don't change
- Decompose by sub-domain pattern:
  - Domain-Driven Design has subdomains and bounded contexts which are useful when modeling microservice architectures
  - Multiple domain models, each with a specific scope specific to the area of concern
  - A bounded context is the scope of a subdomain model, and each one corresponds with a service or set of services

> A class should have only one reason to change - Robert C. Martin

> The classes in a package should be closed together against the same kinds of changes. A change that affects a package affects all the classes in that package. - Robert C. Martin

- Obstacles to decomposition:
  - Network latency: too many round trips
    - Can be mitigated by batch processing or combining services
  - Reduced availability
    - Services rely on one another and if one goes down it affects the others
    - Asynchronous messaging can reduce coupling
  - Data consistency across services
    - Operations may need to update multiple services at once
    - Sagas can deal with this using messaging
  - Consistent view of the data
    - Operations aren't atomic so states between services may not be up to date
    - Prevented by consolidating services into a single service
  - God classes
    - Performs much of the activity of a service
    - Each service has a candidate class that is the primary object of concern for the service
    - To mitigate, each service should have it's own version of each model with only information relevant to the service context
    - Translation of models between services can often happen in the API Gateway

> If you need to update some data atomically, then it must reside within a single service, which can be an obstacle to decomposition.

- When writing the API for your system, start by assigning each system operation to a service

### Articles and Resources

- [O’Reilly Software Architecture Conference](https://conferences.oreilly.com/software-architecture)
- [SATURN conference](https://resources.sei.cmu.edu/news-events/events/saturn/)
- [WikiQuote, _Software Architecture_](https://en.wikiquote.org/wiki/Software_architecture)
- [Software Engineering Institute](http://www.sei.cmu.edu/)
- [_Architectural Blueprints: The ‘4+1’ View Model of Software Architecture_](www.cs.ubc.ca/~gregor/teaching/papers/4+1view-architecture.pdf)
- [_An Introduction to Software Architecture_](https://www.cs.cmu.edu/afs/cs/project/able/ftp/intro_softarch/intro_softarch.pdf)
- [_Pattern: Monolithic Architecture_](https://microservices.io/patterns/monolithic.html)
- [_Pattern: Microservice Architecture_](https://microservices.io/patterns/microservices.html)
- [Wikipedia, _Loose Coupling_](https://en.wikipedia.org/wiki/Loose_coupling)
- [_Pattern: Decompose by Business Capability_](http://microservices.io/patterns/decomposition/decompose-by-business-capability.html)
- [_Pattern: Decompose by Subdomain_](http://microservices.io/patterns/decomposition/decompose-by-subdomain.html)
- [_PrinciplesOfOod_](http://butunclebob.com/ArticleS.UncleBob.PrinciplesOfOod)
- [Cunningham & Cunningham Consultancy Wiki, _God Class_](http://wiki.c2.com/?GodClass)

### Books

- [_Applying UML and Patterns_, Craig Larman](https://www.craiglarman.com/wiki/index.php?title=Book_Applying_UML_and_Patterns)
- _Domain-Driven Design_, Eric Evans
- _Designing Object Oriented C++ Applications Using the Booch Method_, Robert C. Martin

## Chapter 3 - Interprocess communication in a microservice architecture

- Several interaction styles, each with two dimensions
  - First dimension:
    - _One-to-one_
    - _One-to-many_
  - Second dimension:
    - _Synchronous_
    - \_Asynchronous
- Some _one-to-one_ interactions:
  - _Request/response_ - client sends a message and waits for a response and expects a quick response
  - _Async request/response_ - client sends a message and waits for a response but does not block
  - _One-way notifications_ - client sends a message and doesn't wait for a response
- Some _one-to-many_ interactions:
  - _Pub/sub_ - client sends a message and does not wait for responses
  - _Pub/async response_ - client sends a message and waits for an amount of responses (one or more)

> A service's API is a contract between the service and its clients.

- _Interface definition language_ (IDL)
  - Important way to explicitly state the contract between services
  - Helps prevent runtime errors
- Some suggest an API-first design, where the API contract is negotiated before the code is written
- In microservice architectures, it's difficult to update contracts, but there are ways to mitigate the difficulty
- Semantic versioning
  - `MAJOR.MINOR.PATCH`
    - `MAJOR` = breaking change
    - `MINOR` = non-breaking enhancement
    - `PATCH` = non-breaking bug fix
  - Strive to only make backwards-compatible changes, `MINOR` and `PATCH`
  - Robustness principle: "Be conservative in what you do, be liberal in what you accept from others"
    - Clients should ignore extra properties and attributes
    - Services should add sensible defaults when a property or attribute is not present
  - When making breaking `MAJOR` updates:
    - Continue to support the previous version for a time
    - If using REST, you can embed the version in the URL `example.com/v1/path` and `example.com/v2/path` or set the version number in the MIME type like `application/vnd.example.resource+json; version=1`
- Use a cross-language message format
- Text-based message formats
  - JSON, XML, YAML
  - Human-readable and self-describing
  - Easy to make backwards compatible changes
  - Often have well-defined specifications like the XML and JSON schemas
  - Verbose (positive and a negative)
  - Less efficient than alternative formats
- Binary message formats
  - Protocol Buffers, Avro, Thrift, MessagePack
  - Compiler generates code for serializing and deserializing messages
  - Requires an API-first design
  - Can have varying degrees of flexibility for change
- Remove Procedure Invocation (RPI) pattern
  1. Client invokes a _proxy interface_
  2. _RPI proxy_ implements the _proxy interface_ and makes a request to the service
  3. _RPI server_ accepts the requests and uses the _service interface_ to handle it
  4. _RPI server_ sends back a reply to the _RPI proxy_

#### Synchronous messaging

> REST provides a set of architectural constraints that, when applied as a whole, emphasizes scalability of component interactions, generality of interfaces, independent deployment of components, and intermediary components to reduce interaction latency, enforce security, and encapsulate legacy systems.

- REST
  - Richardson maturity model for REST
    - _Level 0_: A single `POST` endpoint which handles multiple actions
    - _Level 1_: Multiple `POST` endpoints, one for each resource, that handle multiple actions
    - _Level 2_: Uses HTTP verbs like `POST`, `GET`, and `PUT`
    - _Level 3_: Same as _level 2_, but `GET` responses include links to actions
      - Based on Hypertext As the Engine of Application State (HATEOAS) principle
  - Open API Specification is the IDL for REST (evolved from Swagger)
  - REST can make it difficult to be efficient by demanding that multiple resources be retrieved via multiple requests
    - GraphQL and Falcor are an alternative method that allows for these batch requests
    - There are mitigation strategies like `?expand=resource`
  - REST also makes it difficult to associate verbs with operations at times
    - Example from Nasdaq: Need to `GET` the odds for the selection, which necessitates putting the selection in the query, rather than a body, due to the limitations of `GET`
  - REST requires usage of a _service discovery mechanism_ or to know about URLs ahead of time
- gRPC (Google Remote Procedure Call)
  - binary message-based
  - Uses Protocol Buffers, which are tagged and allow for backwards-compatibility
  - Is more difficult to consume from a JavaScript client than REST APIs
  - Requires support of HTTP/2
- Request failures should not bring the system down
  - Use a series of mitigation methods:
    - Use _network timeouts_ to ensure requests always end
    - Limit concurrent requests to a service
      - When the limit is reached, new requests should fail instead of attempting to connect
    - If the number of failures is too great, trip a _circuit breaker_ and fail all future requests
- Service discovery can be handled through a number of ways, including self registration and client-side discovery
  - Services register themselves with a centralized registration service
  - Clients request a list of current services from the registration service
  - Eureka is a service registry and Spring Cloud is an example of a client that handles the service discovery and selection
  - Handles mixed environments, like Kubernetes mixed with traditional deployments
  - Requires a service discovery library in every language used, or for you to write your own
  - Must maintain a registry service
- Service discovery can also be handled by platforms like Kubernetes
  - Combines two patterns, _3rd party registration pattern_ and _Server-side discovery pattern_
  - Only supports discovery of services deployed to the platform
  - **Recommended by the author whenever possible**

#### Asynchronous messaging

- Services can use a _message broker_ to communicate
  - Some architectures omit the broker in a _brokerless_ architecture
- Clients do not wait for a response from the server

### Articles and Resources

- [_How to Design Great APIs with API-First Design_, ProgrammableWeb.com](http://www.programmableweb.com/news/how-to-design-great-apis-api-first-design-and-raml/how-to/2015/07/10)
- [Semantic Versioning (semver) specification](http://semver.org/)
- [_Robustness principle_, Wikipedia.org](https://en.wikipedia.org/wiki/Robustness_principle)
- [_XML Schema_, W3.org](http://www.w3.org/XML/Schema)
- [JSON Schema standard](http://json-schema.org/)
- [Protocol Buffers, Google](https://developers.google.com/protocol-buffers/docs/overview)
- [Avro, Apache](https://avro.apache.org/)
- [_Schema evolution in Avro, Protocol Buffers and Thrift_, Martin Kleppmann](https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html)
- [_Pattern: Remove Procedure Invocation_, Microservices.io](http://microservices.io/patterns/communication-style/messaging.html)
- [_Representational state transfer_, Wikipedia.org](https://en.wikipedia.org/wiki/Representational_state_transfer)
- [_Architectural Styles and the Design of Network-based Software Architectures_, University of California, Irvine](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm)
- [_REST APIs must be hypertext-driven_, Roy Fielding](https://roy.gbiv.com/untangled/2008/rest-apis-must-be-hypertext-driven)
- [_Richardson Maturity Model_, Martin Fowler](https://martinfowler.com/articles/richardsonMaturityModel.html)
- [_Advantages of (Also) Using HATEOAS in RESTFul APIs_, InfoQ.com](https://www.infoq.com/news/2009/04/hateoas-restful-api-advantages/)
- [Open API Specification](http://www.openapis.org/)
- <http://graphql.org/>
- [Netflix Falcor](http://netflix.github.io/falcor/)
- <http://www.grpc.io/>
- [_Remote procedure call_, Wikipedia.org](https://en.wikipedia.org/wiki/Remote_procedure_call)
- [_Pattern: Circuit Breaker_, Microservices.io](http://microservices.io/patterns/reliability/circuit-breaker.html)
- [_Fault Tolerance in a High Volume, Distributed System_, Netflix](http://techblog.netflix.com/2012/02/fault-tolerance-in-high-volume.html)
- [Netflix Hystrix](https://github.com/Netflix/Hystrix)
- [Polly .NET Library](https://github.com/App-vNext/Polly)
- [_Pattern: Self registration_, Microservices.io](http://microservices.io/patterns/self-registration.html)
- [_Pattern: Client-side discovery_, Microservices.io](http://microservices.io/patterns/client-side-discovery.html)
- [_Pattern: 3rd party registration_, Microservices.io](http://microservices.io/patterns/3rd-party-registration.html)
- [_Pattern: Server-side discovery_, Microservices.io](http://microservices.io/patterns/server-side-discovery.html)
- [_Pattern: Messaging_, Microservices.io](http://microservices.io/patterns/communication-style/messaging.html)
- [_Message_, EnterpriseIntegrationPatterns.com](https://www.enterpriseintegrationpatterns.com/Message.html)
- [_MessageChannel_, EnterpriseIntegrationPatterns.com](http://www.enterpriseintegrationpatterns.com/MessageChannel.html)
- [_PointToPointChannel_, EnterpriseIntegrationPatterns.com](http://www.enterpriseintegrationpatterns.com/PointToPointChannel.html)
- [_PublishSubscribeChannel_, EnterpriseIntegrationPatterns.com](http://www.enterpriseintegrationpatterns.com/PublishSubscribeChannel.html)

### Books

## Chapter 4 - Managing transactions with sagas

- Microservice transactions can be managed with sagas
- Sagas are `ACD` (missing `Isolation`)
  - Countermeasures can be used to mitigate this loss
  - A saga is defined per system operation
  - Sequence of microservice-level transactions
- Traditionally, distributed transactions have been used
  - Like Open XA with a two-phase commit (2PC)
  - NoSQL databases often don't support them
    - [MongoDb added distributed transactions in 2019](https://www.mongodb.com/press/mongodb-42-adds-distributed-transactions-field-level-encryption-updated-kubernetes-operator-and-more-to-the-leading-modern-general-purpose-database)
  - Modern message brokers don't support them
  - Form of synchronous IPC
- Sagas are rolled back using _compensating transactions_
  - Seems to be easier with event sourcing
  - Read-only transactions do not need these compensations
  - Transactions that always succeed do not need these compensations
- Types of saga transactions
  - _Compensatable_: can fail
  - _Pivot_: followed by steps that cannot fail
  - _Retriable_: always succeed
- Two types of saga coordination:
  - _Choreography_: distributed decision making
  - _Orchestration_: centralized decision authority
- _Correlation ids_ can be used to coordinate transactions between microservices in a choreography-based saga
- Because sagas are not _isolated_, they can result in anomalies:
  1. _Lost updates_: one saga overwrites something written by another
  2. _Dirty reads_: one saga reads an incomplete update
  3. _Fuzzy/nonrepeatable reads_: two sagas read data but get different results
- One countermeasure is pending states like `CANCELLATION_PENDING` or `CREATION_PENDING`
- The two types of saga transactions listed above are not the only two, there's also
  - _Pivot transaction_: A "point of no return" for a saga that cannot be compensated or rolled back
- Eventuate Tram is a saga management framework written by the author

### Types of Countermeasures

#### Semantic lock

Setting a "flag" on any record touched by the saga specifying that it is not yet done (or "committed")

- The `*_PENDING` states
- Can be cleared by successful completion of the saga or a compensating transaction
- Must add logic to prevent and mitigate deadlocks
- Requires participating services to know about the states

#### Commutative updates

A commutative transaction is a transaction that can be executed in any order

#### Pessimistic view

Order the transactions to minimize risk from a dirty read

#### Reread value

Reread a record before updating it once business logic has been executed on it. See the [Optimistic Offline Lock pattern](https://martinfowler.com/eaaCatalog/optimisticOfflineLock.html)

#### Version file

Records the operations performed on a record, which allows them to be reordered and makes commutative operations out of all transactions.

#### By value

Utilize sagas or distributed transactions base on business risk. Higher-risk scenarios should use distributed transactions and lower-risk can instead utilize sagas.

### Articles and Resources

- [_Starbucks Does Not Use Two-Phase Commit_, Gregor's Ramblings](https://www.enterpriseintegrationpatterns.com/ramblings/18_starbucks.html)
- [_Open XA_, Wikipedia](https://en.wikipedia.org/wiki/X/Open_XA)
- [_CAP Theorem_, Wikipedia](https://en.wikipedia.org/wiki/CAP_theorem)
- [_Pattern: Saga_, Microservices.io](http://microservices.io/patterns/data/saga.html)
- [_Transaction Isolation Levels_, MySQL](https://dev.mysql.com/doc/refman/5.7/en/innodb-transaction-isolation-levels.html)
- [_Semantic ACID properties in multidatabases using remote procedure calls and update propagations_, ACM Digital Library](https://dl.acm.org/citation.cfm?id=284472.284478)
- [_Optimistic Offline Lock_, MartinFowler.com](https://martinfowler.com/eaaCatalog/optimisticOfflineLock.html)
- [_About Eventuate Tram_, eventuate.io](https://eventuate.io/abouteventuatetram.html)

## Chapter 5 - Designing business logic in a microservice architecture

### Notes

- Two methods for organizing a microservice's business logic
  - *Transaction script pattern* or procedural code
  - *Domain model pattern* or object-oriented code

#### Transaction script pattern

- "Organize the business logic as a collection of procedural transaction scripts, one for each type of request"
- Methods called _transaction scripts_ handle requests
- Classes are split into state holders and access objects (DTO and DAO models)
  - DTO models are POCOs and don't do much
  - DAO models don't have much data but perform the actions for the DTOs
- Usually coordinated in a `*Service` classes with a method per request type
- Best for simple business logic
- Continuously grow and get harder to maintain

#### Domain model pattern

- "Organize the business logic as an object model consisting of classes that have state and behavior"
- Classes often have both state and behavior
- Still has `*Service` class, but the logic within is simple and most of the complexity is passed into the domain models
- Classes closely model the real-world
- Classes are generally easier to test because they are simpler

#### Domain-Driven Design

- A refinement on object-oriented design focused on complex business logic
- Types of DDD concepts
  - _Entities_ have persistent identities
    - Two entities with the same property values are still different entities
  - _Value objects_ are collections of values
    - Two value objects with the same property values are considered the same
  - _Factories_ are objects or methods that implement complex construction logic
    - Could be static methods on the class
  - _Repositories_ handle the database access logic
  - _Services_ implement business logic that doesn't belong in an entity or a value object
  - _Aggregates_ are a set of domain objects with a boundary, letting them be treated as a single unit
    - "Organize a domain model as a collection of aggregates, each of which is a graph of objects that can be treated as a unit"
    - Have a single _root entity_ with additional entities and value objects
    - Updates to an aggregate must be performed on the root entity
  - _Invariants_ are business rules that must always be enforced

#### Aggregate Rules

1. "Reference only the aggregate root"
2. "Inter-aggregate references must use primary keys"
3. "One transaction create or updates on aggregate"

#### Domain events

- "...a domain event is something that has happened to an aggregate" and is a class in the domain model

### Articles and Resources

- [_Patterns of Enterprise Application Architecture_ by Martin Fowler](https://www.martinfowler.com/books/eaa.html)
- [_Domain-Driven Design: Tackling Complexity in the Heart of Software_ by Eric Evans](https://www.betterworldbooks.com/product/detail/Domain-Driven-Design---Tackling-Complexity-in-the-Heart-of-Software-9780321125217)
- [_What's New in Spring Data Release Ingalls?_, Spring.io](https://spring.io/blog/2017/01/30/what-s-new-in-spring-data-release-ingalls)

## Chapter 6 - Developing business logic with event sourcing

> Note: Graham was bursting at the seams for the discussion this week. We'll see what has him so riled up

- The downside to returning a list of domain events from a domain model modification is that there is no guarantee that the events will be published; they could be ignored by the developer
- Event sourcing, on the other hand, guarantees that an event will be published after a modification or creation of an aggregate

> Event sourcing: Persist an aggregate as a sequence of domain events that represent state changes

- Event sourcing has several perks:
  - History is preserved
  - Domain events are published every time
- Event sourcing has several drawbacks
  - Has a steeper learning curve
  - Querying can be more difficult
- _Object-Relational impedance mismatch_ is when there is a "fundamental conceptual mismatch between the tabular relational schema and the graph structure of a rich domain model with its complex relationships"

> Note: The author mentions that "developers must implement [a history] mechanism themselves" when working with traditional persistance but that isn't exactly true. Many have history including MSSQL

### Articles and Resources

- [_Pattern: Event Sourcing_, Microservices.io](http://microservices.io/patterns/data/event-sourcing.html)
- [_The Vietnam of Computer Science_, Ted Neward](http://blogs.tedneward.com/post/the-vietnam-of-computer-science/)
  - [August 23 Mirror](https://web.archive.org/web/20220823105749/http://blogs.tedneward.com/post/the-vietnam-of-computer-science/)
- [_Introduce Parameter Object_, Refactoring.com](https://refactoring.com/catalog/introduceParameterObject.html)
- [_Memento pattern_, WikiPedia](https://en.wikipedia.org/wiki/Memento_pattern)

## Chapter 7

## Chapter 8

## Chapter 9 - Testing microservices (part 1)

- Chapter begins with a general overview of testing. Includes types of testing, stages of testing, and pipelines
- Rather than the 3-A strategy (Arrange/Assert/Act) it utilizes a four step strategy (Setup/Exercise/Verify/Teardown)
  - Odd that the author chose to make this divergence from such a popular standard
  - Apparently introduced in the book [__xUnit Test Patterns__ by Gerard Meszaros](http://xunitpatterns.com/index.html)
- Asserts that there are four types of tests
  1. Unit tests
  2. Integration tests
  3. Component tests
  4. End-to-end tests
- Uniquely, it uses *component* tests and *end-to-end*, rather than a single *acceptance* test category
  - *Component* tests are acceptance tests for a single component
  - *End-to-end* tests are acceptance tests for the application as a whole

### Consumer-driven contract testing

- When it is necessary to test the communication between to tests, it's preferable to use *consumer-driven contract testing*
  1. Consumer submits tests to some provider using a contract language (e.g., [Pact](https://pact.io/))
  2. Provider compiles tests using the contract(s)
  3. Provider publishes the produced tests to a repository (e.g., Maven, NuGet)
  4. Consumer creates tests for the consumer code using the published test package 
- Frameworks exist for most languages, but **Pact** appears to be the crowd favorite framework
  - [Pact-Net](https://github.com/pact-foundation/pact-net)
    - Uses a shared Rust backend, meaning compatibility is lacking
      - Must use an actual HTTP server, rather than the .NET testing apparatus
      - Does not support x86 systems or ARM architectures
  - [Microsoft's Code with Engineering Playbook: CDC Testing](https://microsoft.github.io/code-with-engineering-playbook/automated-testing/cdc-testing/)
- Contract testing is preferable to schema-based testing (e.g., OpenAPI, JSON schema) because they aren't dynamic enough
- Contracts do not test the provider's business logic, rather that the responses are the proper format (e.g., HTTP method, path, headers, etc.)

### Articles and Resources

- [_Testing Trends for 2018_, Sauce Labs](https://saucelabs.com/assets/3NT3Y0vhCKyVZQjbNCWvjo/204c6fa91deb8adb18813e33d884a8a9/sauce-labs-state-of-testing-2018.pdf)
- [_Test case_, Wikipedia](https://en.wikipedia.org/wiki/Test_case)
- [_Four-Phase Test_, xUnit Patterns](http://xunitpatterns.com/Four%20Phase%20Test.html)
- [_TestPyramid_, Martin Fowler](https://martinfowler.com/bliki/TestPyramid.html)
- [_Pattern: Service Integration Contract Test_, Microservices.io](https://microservices.io/patterns/testing/service-integration-contract-test.html)
- [_Pattern: Consumer-side contract test_, Microservices.io](https://microservices.io/patterns/testing/consumer-side-contract-test.html)
  - Not a very useful link...
- [_Spring Cloud Contract_, Spring.io](https://cloud.spring.io/spring-cloud-contract/reference/html/documentation-overview.html#contract-documentation)
- [_Pact Foundation_, Github](https://github.com/pact-foundation)
- [_UnitTest_, Martin Fowler](https://martinfowler.com/bliki/UnitTest.html)
- [_Eventuate Tram Sagas_, GitHub](https://github.com/eventuate-tram/eventuate-tram-sagas)
- [_MockMvc vs End-toEnd Tests_, Spring.io](https://docs.spring.io/spring-framework/docs/current/reference/html/testing.html#spring-mvc-test-vs-end-to-end-integration-tests)
- [_Rest-assured_, GitHub](https://github.com/rest-assured/rest-assured/wiki/Spring#spring-mock-mvc-module)
