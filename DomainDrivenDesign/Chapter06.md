# Chapter 06 - The Life Cycle of a Domain Object

## Quiz

### 1. What is an _invariant_?

A. An object that does not change once it is created
B. A property of an object that cannot change once initialized
C. A domain rule that must be enforced for an entity or group of entities
D. A constant variable that can be used by multiple entities similar to a singleton

### 2. What is the primary purpose of an _aggregate root_?

A. To hold direct references to all objects in the aggregate
B. To serve as an entrypoint to the aggregate
C. To be the only object from the aggregate that external entities can hold a reference to
D. Both `B` and `C`

### 3. True or False: Entities within the bounds of the aggregate (and not the root) should maintain their global identity

### 4. True or False: An entity within an aggregate can hold a reference to another aggregate root

### 5. Factories are part of which layer of an application?

A. Domain
B. Application/Client
C. UI
D. Repository/Data

### Answers

<details>
<summary>Answers</summary>

1. C
2. D
3. `False` - entities within an aggregate (and not the root) can use local identities, specific to the aggregate
4. `True`
5. A

</details>

## Notes

When considering the life cycle of an object, Evans posits two major challenges,

1. Maintaining integrity though the life cycle of the object
2. Preventing the complexity of life cycle management from overwhelming the model

This chapter introduces the concepts of _Aggregates_, _Factories_, and _Repositories_. _Aggregates_ are roughly the boundaries of scope in which a particular life cycle needs to be maintained, and _Factories_ and _Repositories_ are a means to access those life cycles.

### Aggregates

One of the primary issues with a highly interconnected model comes when you need to remove a particular instance from the application. In the book, Evans uses the example of removing a `Person` object from a database; it's hard to know whether to delete the associated models as well. In the address example, it is possible that the address could be shared between multiple `Person` entities, and by deleting an `Address` for one `Person` when they are deleted, that may also remove it for the other person.

If you chose not to remove those addresses when a use is deleted, then you end up with a large number of orphaned records. You can clean this up using an automated process, but as Evans points out, this is an indication of a modeling issue.

The core issue at hand is where do you stop deleting objects? "How do we know where an object made up of other objects begins and ends?" Enter the _Aggregates_.

An *Aggregate* is "a cluster of associated objects that we treat as a unit for the purposes of data changes." These aggregates are centered around a "Root" object which is then surrounded by a bounds. In this book, this is demonstrated using the example of a `Car` which has `Tire` objects associated to it. Users likely will never need to query for or know about unique tires (if they had any unique identifiers to begin with), and once they are removed from the context of the `Car`, they likely lose all importance (depending on the system; a system concerned with `Tires` specifically would care, but that's not what we're looking at here).

By locking the `Tire` entities within the scope of the `Car`, no outside sources should be able to access a `Tire` without going through a car first.

> ...only _AGGREGATE_ roots can be obtained directly with database queries. All other objects must be found by traversal of associations.

Essentially, the _Aggregates_ should be concerned with their updating, deletion, and modification. If the root is deleted, then all items within the scope are deleted as well. If something outside of the scope needs a reference to the inner objects, then references should be passed out for single-use only (unless they are Value Objects, in which case they lose their context and are new copies anyway).
