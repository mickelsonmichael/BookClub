# Microservices Patterns - Discussions

## Chapter 1

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

## Chapter 2

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

### Books

- [_Applying UML and Patterns_, Craig Larman](https://www.craiglarman.com/wiki/index.php?title=Book_Applying_UML_and_Patterns)
