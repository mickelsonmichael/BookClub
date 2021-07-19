# Designing a programming interface

## 3.1 Introducting REST APIs

- Communication with REST API happens via HTTP.
- The user connects to the server and sends an HTTP Request; the server replies with an HTTP Response.
- An HTTP request has:
    - an **HTTP Method** (E.g. GET)
    - a **Path** (E.g. /products/P123)
    - (optionally) a **Body**
- An HTTP response has:
    - an **HTTP Status Code** (E.g. 200)
    - a **Reason Phrase** (E.g. OK)
    - a **Body** (optional) (E.g. { "productId": "P123", ... })

The HTTP method represents the desired action; the path defines a resource, which represents the functional concept.

E.G. I want to *get* the details of *the product whose ID is P123* -> GET /products/P123

# 3.2 Transposing API goals into a REST API

- A goal is transposed into a resource and action pair.
- Resources are identified by paths.
- Process:
    - Identify resources and relations.
    - Identify actions, parameters and returns.
    - Design resource paths.
    - Represent actions with HTTP.
- From the example of an admin user managing a catalog, we identify two resources: catalog and product, with a "contains many" relationship.
- When we tranlsate our goals, if we are acting on a resource we don't need to include that resource as an input parameter.
- When picking resource paths, they must be unique; but choose user-friendly names.
- Choose paths that best represent the relationship between resources
    - /catalog or /products
    - /product-p123 or /catalog/p123 or /products/p123
- The convention is to represent a collection as the plural of the item's noun, rather than introducting a second term. E.g. /products and /products/p123
- Likewise, nest sub-resouces: /resource/{resourceId}/sub-resouce/{sub-resourceId}
- Map actions to HTTP methods:
    - GET: Get the specified resource
    - POST: Create a new resource in the specified collection
    - DELETE: Deletes the specified resource
    - PATCH: Partially updates the specified resource
    - PUT: Replaces the specified resource (or creates it if it doesn't exist)
- These actions are seen from the user's perspecitive. E.g. Delete doesn't necessarily mean the resource is deleted, it might just be flagged as inactive.
- As a fallback, POST can be used as an "action resource".
- Add parameters in the path (to specify the resource), in the query string (for GET or DELETE inputs) or the body (for POST, PUT and PATCH inputs).

# 3.3 Designing the API's data

- Start by listing all properties of the concept.
- Then for each property, ask:
    - Is it the consumer's business?
    - Is it really needed?
    - Can the consumer understand it?
- Each property must specify:
    - A name (the more descriptive the better)
    - A portable type (string, number, boolean, date, array, object)
    - Whether it is required
    - (Optional) Description
- The same concept can appear in different contexts. Tailor each response to the context.
- Desigining parameters follows the same approach.
- Check that users have access to the source of the parameter

# 3.4 Striking a balance when facing design challenges

- Action resources can perform actions on resources. E.g. "/check-out-cart". These do not conform to the REST model.
- Ask the following questions:
    - What are we trying to represent?
    - What happens in this use case?
- Example: Checking out a cart is creating an order, so "POST /orders" would be an acceptable alternative representation.
- *Sometimes you might not be really satisfied or even a bit dissapointed... this is totally normal.*

# 3.5 Understanding why REST matters for the design of any API

- REST is standardized and therefore abstracts away service implementation.
- REST strives to work for distributed systems:
    - Efficient
    - Scalable
    - Reliable
    - Reusable
- It needs the following:
    - Client/server separation - Server must not delegate responsibility to the client
    - Statelessness - No session, requests must be independently executable
    - Cacheability - Responses must specify if they can be cached (and for how long)
    - Layered system - Server abstracts away inner workings
    - Code on demand (optional) - Server can response with client-executable code (E.g. JavaScript)
    - Uniform interface - All interaction must be through actions taken on resources.
