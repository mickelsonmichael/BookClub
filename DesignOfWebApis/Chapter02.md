# Designing an API for its users

The design of an API must make sense from the perspective of its users.

## 2.1 The right perspective for designing everyday user interfaces

- Focus on what each user is trying to achieve.
- Microwave Oven control panel example:
    - Issues:
        - Exposes inner workings.
        - Doesn't mention heating food.
        - Doesn't provide utility functions.
    - Fixes:
        - Choose labels from the user's perspective.
        - Provide utility function to hide inner workings.

## 2.2 Designing software's interfaces

- Consider an API as the control panel for your software.
- It expresses the user goals that can be achieved.
- Don't require users to write complex logic with potential for bugs.

## 2.3 Identifting an API's goals

- Create a table that helps to identify the goals.
- Start by considering the following questions:
    - What can users do? (E.g. They want to cook food.)
    - How do they do it? (E.g. They heat it at a given power for a given duration.)
- The "what" is broken down into several "how" steps. Each "how" is a goal of the API.
- Online Shopping example:
    - Whats: "Buy Products"
    - Hows: "They add products to the cart." and "Then they check out."
- Then add columns that identify the inputs and outputs of each step.
- Identify any missing goals by considering where these inputs come from and how the outputs are used.
    - Example from input: Where does the *product* come from? -> add a new "what" for searching for products.
    - Example from output: What is done with the *order*? -> add a new "what" for managing orders. 
- Then add a column for "who" to identify different types of users.
    - "User" could mean an end user, the consumer application, or their roles or profiles.

Use a goal canvas:

|Whos               |Whats              |Hows               |Inputs (source)            |Outputs (usage)    |Goals                              |
|-------------------|-------------------|-------------------|---------------------------|-------------------|-----------------------------------|
|Who are the users? |What can they do?  |How do they do it? |What do they need?         |What do they get?  |Reformualte how + inputs + outputs |
|                   |                   |                   |Where does it come from?   |How is it used?    |                                   |

- This is high-level view, don't talk about fine-grained data and errors.
- Start small, don't try to cover all variations of a complex process
- Iterate!

## 2.4 Avoiding the provider's perspective when designing APIs

Conway's Law: *Any organization that designs a system will produce a design whose structure is a copy of the organization's communication structure.*

- Avoid data influences
    - Be wary if the API structure matches the database structure
- Avoid business-logic influences
    - Combine functions into single actions that express user's goals (E.g. "Update Customer Address" rather than "List Customer's Addresses" + "Update Address Status" + "Add Address")
    - Doing this keeps it simple and avoids mis-use of the API.
- Avoid architecture influences
    - Combine data from multiple back-end systems.
- Avoid organizational influences
    - Hide internal steps in the process.

Continually look for these influences creeping in to the design.








