# Chapter 09 - Going deeper into the `DbContext`

## 9.1 Overview of the DbContext class's properties

- Several properties that can be useful for database management on `DbContext`
  - `DbContext.ChangeTracker`
  - `DbContext.Database`
  - `DbContext.Model`

## 9.2 Understanding how EF Core tracks changes

Every entity has a `State` that tracks whether an entity is:

- `EntityState.Added` - new entity to the db
- `EntityState.Unchanged` - existing entity in the db, no changes
- `EntityState.Modified` - existing entity in the db, has changes
- `EntityState.Deleted` - existing entity in the db, will be deleted
- `EntityState.Detached` - un-tracked entity, may or may not be in the database. Will not be seen by `SaveChanges`

If an entity has `EntityState.Modified`, then each property is checked for its `IsModified` property. This ensures that the most efficient query can be built.

```c#
var isModified = context.Entry(entity).Property("PropertyName").IsModified;
```

## 9.3 Details on every command that changes an entity's State

See [Issue 4424](https://github.com/aspnet/EntityFramework/issues/4424) for more information on this section. EF Core now has a more intelligent method of determining the `EntityState` of an object.

### 9.3.1 The Add command - inserting a new row in the database

When an entity is added, EF Core needs to decide what to do with its relationships

- If it is not tracked => `EntityState.Added`
- If it _is_ tracked => `EntityState.Modified`
  - Modifies the necessary foreign keys on the modified object

There are `async` versions of the `Add` commands, but they are not generally used and are more for entities with value generators.

### 9.3.2 The Remove command - deleting a row from the database

When an entity is removed, the behavior depends on the primary key:

- Database-generated column && not the default value => `EntityState.Modified` || `EntityState.Unchanged`
  - If the related entity doesn't have a foreign key (e.g. is not a dependent entity, like an `Author` with books), then no changes are necessary and the state is `Unchanged`
- Not a database-generated column || key is default value => `EntityState.Added`

While the second scenario is set to added, it's entirely possible and probable that the entity state will be changed to `EntityState.Deleted` due to cascading deletes, thus it will never be added to the database in the end.

You can delete an entity pulled down using `AsNoTracking`; EF Core will look for a non-default primary key.

### 9.3.3 Modifying a tracked entity - EF Core's `DetectChanges`

The `DetectChanges` method is called each time you call `SaveChanges`. It will compare the tracking snapshot EF Core generated when you first pulled the entity with the current state and update each of the `IsModified` flags for the properties. **This will only happen for an entity with state `Unchanged`.\***

\*_This may need further verification. The author only mentions this in a caption and nowhere else._

See [this blog post](https://blog.oneunicorn.com/2012/03/10/secrets-of-detectchanges-part-1-what-does-detectchanges-do/) and [this blog post](https://blog.oneunicorn.com/2012/03/11/secrets-of-detectchanges-part-2-when-is-detectchanges-called-automatically/) for more information (these were not provided by the author but found while doing research). The following are some of the methods that call `DetectChanges`

- `DbSet.Find`
- `DbSet.Local`
- `DbSet.Remove`
- `DbSet.Add`
- `DbSet.Attach`
- `DbContext.SaveChanges`
- `DbContext.GetValidationErrors` (EF6)
- `DbContext.Entry`
- `DbChangeTracker.Entries`

### 9.3.4 `INotifyPropertyChanged` entities - a different way of tracking changes

`DetectChanges` can be a slow process, and in performance-sensitive application that may be a deal breaker. Instead you can implement `INotifyPropertyChanged`. Every time you update a property of an Entity, the `SetWithNotify<T>` method will have to be called, leading to a large amount of noisy code.

Additionally, you must use an observable to any navigational properties. EF Core recommends `ObservableHashSet` for performance reasons. The author doesn't really elaborate on this more.

Once you implement the interface, you will also need to set the tracking strategy to `ChangedNotifications`.

```c#
modelBuilder.HasChangeTrackingStrategy(
    ChangeTrackingStrategy.ChangedNotifications
);

// OR

modelBuilder
    .Entity<MyEntity>()
    .HasChangeTrackingStrategy(ChangeTrackingStrategy.ChangedNotifications);
```

These changes lead to a very quick `SaveChanges` method, regardless of how many entities are loaded.

EF Core still takes a tracking snapshot for other reasons but it isn't used by `DetectChanges`.

### 9.3.5 The `Update` method - telling EF Core that everything has changed

When using `Update`, all properties are sent back to the database; they all have their `IsModified` flags set to `true` and the entity is set to `EntityState.Modified`.

When there are navigational properties, they have their state set to

- Database-generated key && not the default value => `EntityState.Modified` || `EntityState.Unchanged`
  - If a foreign key property needs to be changed, then `Modified`, otherwise `Unchanged`
- Not database-generated key || default key value => `EntityState.Added`

### 9.3.6 The `Attach` method - changing an un-tracked entity into a tracked entity

If you attach an entity, it is tracked and **EF Core assumes that its contents matches the current database state**, thus its `State = EntityState.Unchanged`.

For navigational properties

- Database-generated key && not default key value => `EntityState.Unchanged`
- Not database-generated key || default key value => `EntityState.Added`

Shadow properties aren't part of the class, so they are lost in serialization. You must restore any shadow properties, **especially foreign keys**, after `Attach` has been called.

### 9.3.7 Setting the `State` of an entity directly

You can always manually change the `State` of an entity or the `IsModified` flag of a property; the fields are read/write. **If the entity wasn't tracked before you set the `State`, it will be tracked after**.

### 9.3.8 `TrackGraph` - handling disconnected updates with relationships

This is useful in disconnected updates. The graph will traverse all the relational links in the entity and call an action (supplied by you) on each entity it finds.

This section is underwhelming. The author provides a very static and frankly useless example.

## 9.4 Using `ChangeTracker` to detect changes

You can use the change tracker to detect when an entity has been changed and leverage that to perform a number of operations.

The author recommends overwriting all of the `SaveChanges` methods and execute a method that utilizes the `ChangeTracker.Entities` list of entities to check which entities are in a `Modified` state. In the author's example, he modified the entity's updated date each time changes were saved by utilizing an interface.

## 9.5 Using raw SQL commands in EF Core

There are two methods for executing raw SQL

- `FromSql`
- `ExecuteSqlCommand`

Then the `Reload` command and `GetDbConnection` commands that you can leverage as well.

### 9.5.1 `FromSql` - adding raw SQL to an EF Core query

Allows you to call raw SQL commands and retrieve entities from the database

```c#
var books = context.Books
    .FromSql(
        $@"EXECUTE dbo.spFilterOnReviewRank
            @RankFilter = {rankFilterBy}"
    ).ToList();
```

The `FromSql` will utilize the string interpolation and pass the arguments in as parameters to the SQL command, preventing SQL Injection attacks. **However, this only works if you define the interpolated string _inside_ the `FromSql` command**; if you use a variable, then you lose the protection.

If you are loading an entity class, the query must return data for _all_ properties of the type so that the entity can be tracked.

You will need to disable model-level query filters if you want to use things like `ORDER BY`. This can be done by applying `IgnoreQueryFilters()` before the `FromSql` command.

### 9.5.2 `ExecuteSqlCommand` - executing a non-query command

This method is mainly used for updating and deleting from the database.

```c#
int rowsAffected = context.Database
    .ExecuteSqlCommand("UPDATE Books SET IsOpen = 0");
```

The return value is an integer that represents how many rows were updated or deleted.

### 9.5.3 `Reload` - useful after an `ExecuteSqlCommand`

Can be called after the `ExecuteSqlCommand` to update the tracked entities with the new versions.

### 9.5.4 `GetDbConnection` - calling a database access commands

This command can be used to get the `SqlConnection` for the database. That can then be leveraged for a number of reasons, including ADO.NET.

## 9.6 Using `Context.Model` to access EF Core's view of the database

`DbContext.Model` will provide information about the database like table names, column names, and more. This is useful for building queries or commands using raw SQL.

### 9.6.1 Using the `Model` property to build a fast database wipe method

The author then walks through an entire example of how you could leverage the `Model` to delete all the rows in the database without using the `EnsureDeleted` command. This can be more performant for testing. You can get all the code [via this link](http://mng.bz/07vz).

## 9.7 Handling database connection problems

EF Core allows the configuration of an execution method that will determine who many times a transaction is attempted and how to handle certain issues.

### 9.7.1 Handling database transactions with EF Core's execution strategy

You can wrap certain code segments in an `Execute` call to have it be protected by the current execution strategy.

```c#
var strategy = context.Database.CreateExecutionStrategy();

strategy.Execute(() => {
    // do some update here

    context.SaveChange();
});
```

### 9.7.2 Altering or writing your own execution strategy

You are able to create your own execution strategy by implementing `IExeuctionStrategy` and checking out [the `SqlServerExecutionStrategy` method as a guide](https://github.com/dotnet/efcore/blob/main/src/EFCore.SqlServer/Storage/Internal/SqlServerExecutionStrategy.cs). You can then add it to your configuration options for your DbProvider.
