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

### 8.4.3 Creating a value generator to generate a default value dynamically

- You can create a value generator by overriding the `ValueGenerator<T>` class
- Will only set the value if both:
  - `State == Added`
  - The property is the default .NET type (e.g. `default(int)`)
- Override the `GenerateTemporaryValues` method to determine whether the value should be written to the database
- Override the `Next(EntityEntry entry)` method which is called when you add the entity to the DbContext
- Because `Next` is called before the write to the database, no database-generated values, like the primary key, are defined yet
- There is a `NextAsync` version

## 8.5 Sequences - providing numbers in a strict order

- Databases allow for a specific "pool" of numbers, enforcing a uniform progression (e.g. 1,2,3,4)
- Can define the start number and the increment
- Access the number using `NEXT VALUE FOR myEntity.MyProperty` in SQL; will be different for other providers

```c#
modelBuilder.HasSequence<int>("BookNumber") // can provide a schema as a second parameter
    .StartsAt(100)
    .IncrementsBy(5);

modelBuilder.Entity<Book>()
    .Property(b => b.BookId)
    .HasDefaultValueSql("NEXT VALUE FOR dbo.BookNumber");
```

## 8.6 Marking database-generated properties

- May need to mark a property as database-generated
- Usually does not need to happen, but worth knowing in the event you need it
- Three types of columns you can define
  - Generated
  - Added on insert
  - "Normal"
- EF6 has the same system, but Core adds Fluent API calls too

### 8.6.1 Marking a column that's generated on an addition or update

- For letting EF Core know when a column is read-only
- Use the `DatabaseGeneratedAttribute` with an argument `DatabaseGeneratedOption.Computed`

```c#
public class Book
{
    [DatabaseGenerated(DatabaseGeneratedOption.Computed)]
    public int YearOfPublication { get; set; }
}
```

```c#
modelBuilder.Entity<Book>()
    .Property(b => b.YearOfPublication)
    .ValueGeneratedOnAddOrUpdate();
```

### 8.6.2 Marking a column's value as set on insert of a new row

- Two ways to give a column data on insertion
  - Key generation (e.g. `IDENTITY` in SQL)
  - Default constraint if no value is provided
- It is unusual to need to tell EF about identity columns; it usually knows about them
- If you need to, you can use the `DatabaseGeneratedAttribute` again with the argument set to `DatabaseGeneratedOption.Identity`
- You can also use the fluent API

```c#
modelBuilder.Entity<Book>()
    .Property(b => b.YearOfPublication)
    .ValueGeneratedOnAdd(); // notice the missing "OrUpdate"
```

### 8.6.3 Marking a column as "normal"

- When you are generating the value client-side and don't want EF to generate it for you
- Common scenario is when utilizing a `GUID`
- If you use a `GUID` as a primary key, EF core will generate a value if you don't supply one using a internal value generator
- You can turn off this generation implementation using the `DatabaseGeneratedAttribute` and setting the argument to `DatabaseGeneratedOption.None`
- Again, can also use the Fluent API

```c#
modelBuilder.Entity<Book>()
    .Property(b => b.ISBN)
    .ValueGeneratedNever();
```

## 8.7 Handling simultaneous updates - concurrency conflicts

- By default, EF uses Optimistic Concurrency pattern; the newest update overwrites the previous

### 8.7.1 Why do concurrency updates matter

- Clashing information overwriting one another can mean that valuable history is lost
- One design pattern for preventing this loss is using the _event sourcing_ approach; keeping a table of events and the date they occured, then sorting them by date and grabbing the latest

### 8.7.2 EF Core's Concurrency conflict-handling features

- Two methods built in
  - Concurrency token
  - Timestamp
- EF6 has the same features, but they have been re-implemented in Core
- When a conflict is found, a `DbUpdateConcurrencyException` is thrown

#### Detecting a concurrent change via concurrency token

- Mark individual properties as needing protection
- Utilizes the `ConcurrencyCheckAttribute` to denote which properties to protect
- On update, if the value passed does not match the value in the database, the `DbUpdateConcurrencyException` is thrown
- **Important to note that only the property marked as a concurrency token is checked**
- This is done by adding the token to the `WHERE` statement of the `UPDATE`
- The `UPDATE` command returns the number of rows updated, if 0 is returned, then EF Core throws the exception
- Works on any database because it leverages a basic command

#### Detecting a concurrent change via timestamp

- This method is database-specific and may not be available on all
- Uses a unique value provided by the database that's updated whenever the row is inserted or updated
- Add a `byte[]` property and decorate it with the `TimestampAttribute`
- Checks all the properties, not just individual ones this way; if the row has been updated at all the exception will be thrown
- Which version to use depends on your business rules and database

### 8.7.3 Handling a `DbUpdateConcurrencyException`

- Implement a custom save method that catches the `DbUpdateConcurrencyException`, then performs an action with the value
- The author's implementation returns a string with the error message to the user after comparing the values using reflection on the properties

### 8.7.4 The disconnected concurrent update issue

- In web-based scenarios where there may be multiple users making changes over large amounts of time
- Handling of the conflicts may need to be more verbose
- Author again returns an error message after catching the exception, but allows the user to resolve the conflict themselves
