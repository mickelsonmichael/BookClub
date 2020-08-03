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
- For beginners, you can use the `context.Database.ExecuteSqlCommand(string)`

### 8.2.3 Using a registered scalar UDF in your database queries

- You can call the static method before you enumerate your results like any other static method
- Be careful, because if you call `AsEnumerable` or `ToList` _before_ you call your UDF function, the client-side implementation will be used
- The server UDF version will only be utilized if it is before the results are pulled from the database. [See this link](https://docs.microsoft.com/en-us/ef/core/querying/client-eval#explicit-client-evaluation)
- UDFs can be used in filtering, selecting, and more

```c#
// The UdfClass is a custom class containing your static method
var stuff = context.Books.Select(b => new {
    Id = b.BookId,
    Title = b.Title,
    Rating = UdfClass.AverageVotes(b.BookId)
}).ToList();
```

## 8.3 Computed column - a dynamically calculated column value

- Column that is calculated when you read the column
- EF core will read back the column each time a select, update, or insert is performed
- The value is calculated each time, so this can be expensive if the function isn't simple enough

```c#
modelBuilder.Entity<Author>()
    .Property(a => a.NumberOfBooks)
    .HasComputedColumnSql("COUNT()");
```

## 8.4 Setting a default value for a database column

- There are three methods to doing this
- Defaults can only be applied to scalar properties, but they can be backing/shadow
- Defaults will only be applied if they match the CLR default (e.g. if the date == default(DateTime) or int == default(int))
- The defaults won't be applied until `SaveChanges` is called. If you need them before that call, you will have to use C# methods like property initializers
- "**default values happen only on new rows added to the database. They don't apply to updates.**"

### 8.4.1 Adding a constant as a default constraint

- You can use the Fluent API to set the default
- EF Core will read the value when `SaveChanges` is called. **Not before**.
- **This will be a constant value**. Using `HasDefaultValue(DateTime.Now)` will not work; it would instead use the time when the context was configured (application startup)

```c#
modelBuilder.Entity<Book>()
    .Property("PublishedDate")
    .HasDefaultValue(new DateTime(2000, 1, 1));
```

### 8.4.2 Adding an SQL fragment as a default constraint

- You can use SQL functions in place of C# functions for the default
- This can be useful with `CreatedDate` columns; set the property to `private set` and the devs won't be able to accidentally change the values at runtime
- You can't reference another column in the default constraint
- You can also call UDF functions as well

```c#
modelBuilder.Entity<Book>()
    .Property(b => b.CreatedDate)
    .HaDefaultValueSql("GETUTCDATE()");
```
