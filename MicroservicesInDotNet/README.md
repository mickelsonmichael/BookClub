# Microservices in .NET, Second Edition

_by Christian Horsdal Gammelgaard_

- [Microservices in .NET, Second Edition](#microservices-in-net-second-edition)
  - [1: Microservices at a Glance](#1-microservices-at-a-glance)
    - [Microservices: a definition of this new architectural term](#microservices-a-definition-of-this-new-architectural-term)
    - [Single Responsibility Principle](#single-responsibility-principle)
    - [Monolith First](#monolith-first)
    - [Discussion](#discussion)
  - [2: A basic shopping cart microservice](#2-a-basic-shopping-cart-microservice)
  - [3: Deploying a microservice to Kubernetes](#3-deploying-a-microservice-to-kubernetes)
  - [4: Identifying and scoping microservices](#4-identifying-and-scoping-microservices)

## 1: Microservices at a Glance

- "A _microservice_ is a service with one, and only one, very narrowly focused capability that a remote API exposes to the rest of the system."
  - Two kinds of capabilities: _business_ and _technical_
- Each microservices should have it's own dedicated data store
- A microservice doesn't have to be implemented in the same language as all other microservices (e.g. C#, Python, JavaScript)
  - They just need to know how to communicate with each other using some standard like HTTP or gRPC
- [Microservices: a definition of this new architectural term](#microservices-a-definition-of-this-new-architectural-term)
- [Single Responsibility Principle](#single-responsibility-principle)
- **Business capabilities** meet some business requirement, like calculating prices
  - Easily determined when using domain-driven design
- **Technical capabilities** are things multiple microservices may need use and are not the main focus of breaking a system into microservices, and usually evolve out of several microservices having the same technical requirement
- Kubernetes supports rolling deployments as well as other useful patterns for managing microservices
- Changes to a microservices should be backwards compatible when possible to reduce the effect on other, dependent microservices
- A microservice often needs two or more process which can include things like a web server and a database process.
- A small team of around five people should be capable of managing something in the range of 10 to 30 microservices.
- Teams should factor out microservices into smaller microservices when they grow too large
- Operations should be able to determine the health of any microservice with ease
- Microservices are often created from monolith applications, and have a high degree of success when starting life as one
  - [Monolith First](#monolith-first)
- Code reuse is generally frowned upon and incurs costs that aren't always obvious
  - Moving functions outside of a microservice makes it more inconvenient to work with and understand the microservice
  - Shared libraries introduce coupling between services and inflexibility
  - Shared libraries will need to serve several different purposes rather than a single, well-define purpose

### Microservices: a definition of this new architectural term

This is a referenced article by James Lewis and Martin Fowler. It can be accessed for free on [MartinFowler.com](https://martinfowler.com/articles/microservices.html).

- Microservice architectures are "a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API."
- [Microservices Resource Guide](https://martinfowler.com/microservices)
- Not all microservices architectures have the same characteristics, but should have _many_ of the same characteristics
- "A **component** is a unit of software that is independently replaceable and upgradeable."
  - Libraries are in-memory and in-process
  - Services are out-of-process and usually communicate over a network (HTTP, gRPC)
    - Are independently deployable
    - Make it easier to enforce encapsulation and looser coupling
    - Remote calls are more expensive
    - May include multiple processes
- Microservices are organized around business capabilities rather than functional capabilities
- Teams can handle microservices of varying size, but there are two general bounds to the number of microservices a team can handle
  - One service per 12 people (two pizza team)
  - One service per person, or 12 services for 12 people
- Team working on microservices tend to lean towards stronger ownership of the service, rather than handing it off when complete
- Communication should be more intelligent and handle things like routing and business rules
- SOA = Service Oriented Architecture
  - A "dirtier" term, because it has been poorly defined and implemented to this point
  - Some still consider microservices to be SOA
- Because they are decoupled, different languages or databases can be used for each microservice if necessary
- The [TolerantReader](https://martinfowler.com/bliki/TolerantReader.html) and [Consumer-Driven Contracts](https://martinfowler.com/articles/consumerDrivenContracts.html) are often leveraged

### Single Responsibility Principle

This is a referenced article by Robert C. Martin (aka Uncle Bob). It can be access for free on [blog.CleanCoder.com](https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html).

### Monolith First

This is a referenced article by Martin Fowler. It can be accessed for free on [MartinFowler.com](https://martinfowler.com/bliki/MonolithFirst.html)

### Discussion

- [Lerna: Monorepo Deployment](https://lerna.js.org/)
- [Open API Generator](https://openapi-generator.tech/)

## 2: A basic shopping cart microservice

- Rather than grouping the code by function (e.g. `/Models`, `/Controllers`, `/Services`), the author groups them by domain (e.g. `/ShoppingCart`, `/Users`)

## 3: Deploying a microservice to Kubernetes

This chapter is mostly a review for the Nasdaq group. If the LMCU folks would like to add any notes here, please feel free.

## 4: Identifying and scoping microservices

- Three drives for identifying and scoping microservices
  1. Business capabilities
  2. Technical capabilities
  3. Efficiency of work
- Business capabilities contribute towards business goals
- Conway's Law: *Any organization that designs a system (defined broadly) will produce a design whose structure is a copy of the organization's communication structure.*
- Sometimes microservices won't correctly match the domain of the organization, and you can either modify the microservices to match the organization or modify the organization to match the microservices
- Discussion: When the business unit isn't sure about the domain language
