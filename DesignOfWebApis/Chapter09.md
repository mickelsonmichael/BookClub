# Evolving an API Design
- Covers breaking changes, versioning, and extensibility
## 9.1 Designing API Evolutions 
*aka Avoiding Breaking Changes*
1. Input and Output Data
    - Breaking
        - Renaming a property
        - Moving a property
        - Removing a mandatory property
        - Making a mandatory property optional (output)
        - Modifying the property type
        - Modifying the property format
        - Modifying the property characteristics
            - Output includes increasing length of a string or a range of numbers or an array
            - Input includes descreasing length of a string or a range of numbers or an array
        - Modifying the property meaning
        - Adding values to enums (output)
        - Removing values from enums (input)
    - Non-breaking
        - Adding optional properties
            - Can have a mandatory property in an optional object
        - Making an optional property mandatory (output)
        - Making a mandatory property optional (input)
2. Success and Error Feedback
    - Breaking
        - Renaming properties
        - Adding a new enum
    - Non-breaking
        - Modify the HTTP status code
        - *this applies only if the class of the status code remains the same*
3. Goals and Flows
    - Breaking
        - Renaming or Removing goals (endpoints)
    - Non-breaking
        - Adding new goals
4. Security Breaches
    - Breaking: changing authentication
    - When adding, removing or updating goal responses, must revisit scope 
5. Be aware of the invisible interface contract
    - Hyrum's Law
        - *With a sufficient number of users of an API, it does not matter what you promise in the contract:*
        *all observable behaviors of your system will be depended on by somebody*
6. Breaking changes are *usually* only a problem for third party applications
    - internal consumers of the api *usually* can make the changes on the application to match the changes in the API
## 9.2 Versioning an API
- Contrasting API versioning with Implementation versioning
    - Semantic versioning
        - Major.Minor.Patch (1.1.10)
        - Breaking.NonBreaking (2.1)
    - an API version does not always correlate to the implementation version (i.e. 1.1 or 2.2)
    - Consumers usually only care about the breaking changes
- Choosing an API versioning representation
    - Domain *different url*
    - Resource path *same domain, different path*
    - Query Parameter *?version=*
    - Custom Header *Version : 2*
    - Content Type Negotiation *application/vnd.bank.2*
    - Request body
- Choosing API version granularity **not for REST API's**
    - Resource/Path
        - only gives hints about what changed
        - used for independent resources (microservices?)
    - Goal/Operation
        - indicates which goals have changed
        - difficult for the user to know which goal versions will work together
    - Data/message
        - indicates which data/messages have changed
## 9.3 Designing an API with extensibility in mind
- Design to allow the addition of new functionality
- Data
    - objects instead of arrays or value types
    - single property to replace multiple when possible
        - possibly including a code and descriptive text 
    - group similar data into a collection
    - standards
- Interactions 
    - reusability for feedback
    - ignore extra parameters in the request
    - reduce request values to max limit when possible
        - take the request data into consideration
- Flows
    - different use cases
    - choose widely used inputs and outputs
- API
    - create smaller API's
