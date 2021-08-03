# Designing a Straightforward API

__Main purpose is usability__

## 5.1 Designing straightforward representations

- The choices you make with regard to names, data formats, and data can greatly enhance or undermine an API's usability.
- Consider whether the chosen representations make sense and are easily understandable for the consumer, focusing on the consumer's perspective and designing with the user in mind.

### 5.1.1 Choosing crystal-clear names

- To use words that consumers can understand easily.
- Find a short but still clearly understandable name.
- Avoid abbreviations and exposing internal code conventions.
- This is for resources, parameters, properties, JSON schemas, or anything else that needs a name.

### 5.1.2 Choosing easy-to-use data types and formats

- Inadequate data representation can ruin usability
- We care about data types and formats when designing an API because an API is a user interface.
- Developers rely not only on names to understand and use the API, but also on its data.

```yaml
# NOT easy-to-use data
number: 123457
balanceDate: 1534960860
creationDate: 1423267200
type: 1

# easy-to-use data
number: 0001234567
balanceDate: "2018-08-22T18:01:00z"
creationDate: "2015-02-07"
type: "checking"
```

- Developer could easily understand balance `date` instead of a UNIX timestamp. Its ISO 8601 string counterpart "YEAR-MONTH-DAY".
- The type property is supposed to tell if an account is a checking or savings account. While on the not easy-to-use side, its value is a cryptic number
- The string value `0001234567` is easier to use and is not corrupted.
- When choosing data types and format, must be human friendly and, most importantly, always provide an accurate representation. When using a complex format, try to provide just enough information. And, if possible, try to stay understandable without context.

### 5.1.3 Choosing ready-to-use data

```yaml
# NOT ready-to-use data
# GET /accounts/473e2393-a3b3-3941-aa48-d8163ead9ffc
type: 2
balance: 500
overdraftLimit: 100
creationDate: "2015-02-07"

# Ready-to-use data
# GET /accounts/0001234567
type: 2
typeName: "checking" # adding data to clarify cryptic values
balance: 500
currency: "USD" # providing added-value data to ensure consumers have nothing to do on their side
overdraftLimit: 100
safeToSpend: 600
yearsOpen: 3 # replacing data with more relevant added-value data
```

- To clarify such numerical values like `type`, we can provide an additional `typeName` property.
- Provide a ready-to-use `safeToSpend` property in order to avoid the consumer having to do this calculation.
- We can also provide information about the account's currency so consumers know that `balance`, `overdraftLimit`, and `safeToSpend` amounts are in US Dollars (USD).
- Providing static or precalculated added-value data ensures that consumers have almost nothing to do or guess on their side.
- Use a more user-friendly value such as the bank account number, which is unique. That way, the URL to identify a bank account becomes `/accounts/{accountNumber}`. The `/accounts/0001234567` path is still unique and now has a clearer meaning.

## 5.2 Designing straightforward interactions

The type of interaction should be straightforwardd, requiring minimal, understandable, and easy-to-provide inputs and showing helpful error feedback and informative success feedback.

### 5.2.1 Requesting straightforward inputs

The first step of an interaction belongs to the users. It's up to them to provide some inputs to say what they want to do. As API designers, we can give them a hand by designing straightforward inputs like those on the easy-to-use washing machine (Figure 5.6), using what we learned earlier in this chapter.

### 5.2.2 Identifying all possible error feedback

Three different types of error -- malformed, function, and server.

1. __**Malformed request errors**__:
  - The data stream sent by the client to the server didn't follow the rules or the server is unable to interpret a request.
2. __**Functional errors**__:
  - Triggered by the implementation's business rules
  - If something that you expect it to do is awkward, confusing, or impossible.
  - If the "Cancel" button is not clickable then it is a functionality error.
  - The money transfer's amount might exceed the safe-to-spend value.
  - Exceeds the maximum amount the user is allowed to transfer in one day.
  - It may be forbidden to transfer money to an external account from certain internal accounts.
  - Mostly occur when consumers try to create, update, or delete data or trigger actions. They can typically be identified once the API goals canvas is filled in because each goal is fully described from a functional point of view.
3. __**Server Error**__:
  - You send a vlid request and you didn't get what is expected.
  - Can be caused by a down database server or a bug in the implementation.
  - 500 Internal Server
  - On server errors, consumers just need to know that their request could not be processed and that it is not their fault. That's why a single generic server error is sufficient. But identifying possible errors is not enough; we must design an informative representation for each of them.

### 5.2.3 Returning informative error feedback

- Provide as much data as necessary in order to help consumers solve the problems themselves. You could, for instance, provide a regular expression describing the expected data formit in the case of a `BAD_FORMAT` error.
- Providing informative and efficient feedback requires us to describe the problem and provide all needed information in both human- and machine-readable format in order to help the consumer solve the problem themselves (if they can).
- When designing a REST API, this can be done by using the appropriate HTTP status code and straightforward response body. That works for reporting only one error at a time. But what if there are multiple problems?

### 5.2.4 Returning exhaustive error feedback

Grouping multiple errors in one feedback message simplifies an interaction by reducing the number of request/error cycles. But if you are designing a REST API, it means using a generic HTTP status and relying on the response data to provide detailed information about each error. Once all problems are solved, the interaction should end with a success feedback.

### 5.2.5 Returning informative success feedback

So basically, informative success feedbacks provide information about what has happened and also give information that can help during the next steps.

Rules we've identified for designing straightforward interactions:

1. Inputs and outputs must be straightforward
2. All possible errors must be identified
3. Error feedback must explain what the problem is and should help the consumers to solve it themselves.
4. Reporting multiple errors one-by-one should be avoided.
5. Success feedback should provide information about what was done and give information to help for the next steps.

## 5.3 Designing straightforward flows

To use an object or an API, a user might have to chain multiple interactions.

Usability heavily depends on the simplicity of this flow of interactions.

The interaction flow simplified by improving feedback, improving inputs, preventing errors, and even aggregating actions. This interaction flow has become totally straightforward.

### 5.3.1 Building a straightforward goal chain

A chain exists only if its links are connected. When consumers use an API for a specific goal, they must have all the data needed to execute it. Such data may be known by the consumers themselves or can be provided by previous goal outputs.

So the first step towards a straightforward API goal chain is to request simple inputs that can be provided by consumers or another goal in the chain, and return exhaustive and informative error feedback to limit request/error cycles.

Example via Figure 5.14

The list accounts and list beneficiaries golas are pretty straightforward because they do not need inputs and return no errors.

The inputs to transfer money goal are straightforward , but this goal can return many different errors.

Then you must provide informative and exhaustive error feedback in order to help consumers solve the problems they encounter.

This will greatly reduce the number of request/error cycles and avoid artificially extending the API call chain lenght.

### 5.3.2 Preventing errors

Example, addinga  direction indicator helped to prevent an unexpected trip to the ground floor of an elevator.

Preventing errors can make the goal more fluid. You can do this by:

1. Analyzing possible errors to determine added value-data that could prevent them
2. Enhancing the success feedback of existing goals to provide such data or creating new goals to provide such data.

### 5.3.3 Aggregating goals

Putting the floor buttons outside the elevator cabin permitted replacing the "call an elevator" and "select the 16th floor" actions with a single one: "call an elevator to go to the 16th floor". Such aggregations can be useful for optimizing the API goals flow.

### 5.3.4 Designing stateless flows

**Statelessness** is achieved by storing no context on the server between requests (using a session) and only relying on the information provided along with a request to process it.

That ensures that any request can be processed by an instance of an API implementation instead of a specific one holding the session data.

And that also favors the use os the API goals independently and, therefore, facilitates their reuse in different contexts.
