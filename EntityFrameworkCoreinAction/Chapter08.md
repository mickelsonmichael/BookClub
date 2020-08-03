# Chapter 8 - Configuring advanced features and handling concurrency conflicts

## 8.1 Advanced feature - using backing fields with relationships

- Backing fields are essentially ways to add additional validation to a property by preventing the devs from accessing it directly

### 8.1.1 The problem - the book app performance is too slow

- Each time a book is brought up, the system has to average all of the current votes for the reviews. The author points out this could be optimized by stashing that calculation in a property
- If it's in a precalculated state, then we'll have to update it each time a review is added or removed
- We will learn a potentially better method in section 13.4.2, but for now this will do

### 8.1.2 Our solution - `IEnumerable<Review>` property and a backing field

- Change the `ICollection<Review>` to an `IEnumerable<Review>`, which will remove the `Add` and `Remove` methods from the collection and hide the real collection in a backing field
- Create an `AddReview` and `RemoveReview` method to the `Book` entity
- You can name the backing field `_reviews` and EF Core should configure it by convention
- The add and remove methods should update a `CachedVotes` property each time they are called
- You must add some configuration to the fluent API to ensure EF Core will write to the backing field

```c#
entity.Metadata
    .FindNavigation(nameof(Ch07Book.Reviews))
    .SetPropertyAccessMode(PropertyAccessMode.Field);
```

## 8.2 DbFunction - using user-defined functions with EF Core

- UDFs = User-Defined Functions
- Allow you to write SQL code that will be run on the db server; moves a calculation from client-side to server which can be faster since the db can access the rows directly
- _Database scalar function mapping_ (DbFunction) allows you to reference a UDF in the db as if it were a local method in code
- These are very useful for performance tuning
- You must manually add the UDF into the database using an SQL Command
- Declare a method in the application's DbContext or in a separate class with the proper configuration
- You can then use the UDF reference in a query and EF core will convert it into a db call to the SQL query

### 8.2.1 Configuring a scalar user-defined function

- A static method matching the SQL method and registered with EF Core at config time
- The name of the static method should match the UDF, but can be configured otherwise
- The number, type and order of the parameters must match the parameters of the UDF
- _Name of the parameters does not need to match_
- Register in one of two ways:
  - Use the `DbFunctionAttribute` on the method
  - Use the Fluent API

```c#
// DbFunctionAttribute
[DbFunction]
public static double? AverageVotes(int id)
{
    //some code
}

// Fluent API
modelBuilder.HasDbFunction(
    () => WrappingClassName.AverageVotes(default(int))
);
```

- The method in the `HasDbFunction` call isn't actually called, but reflection is used to find the name, return type, and parameters of the method

### 8.2.2 Adding your UDF code to the database

- You must manually add the UDFs to the database with SQL
- You can leverage migrations to do this, but that is discussed in chapter 11
