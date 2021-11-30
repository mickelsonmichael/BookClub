# Identity-based access control

> Read **November 30, 2021**
>
> Presented by **Raghda**
>
> Code by **Michael**

Contents:
- [Raghda's Notes](#raghda)
- [Michael's Notes](#michael)

## Raghda

This chapter talks about alternative ways of organizing permissions in the identity-based access control model.

1. **Identity-based access control (IBAC)** determines what you can do based on who you are. The user performing an API request is first authenticated and then a check is performed to see if that user is authorized to perform the requested action.
2. The approach to simplifying permission management is to collect related users into groups, where permissions are assigned to collections of users.
3. The advantage of groups is that you can assign permissions to groups and be sure that all members of that group have consistent permissions.
4. In UNIX groups, the owner of the file can assign permissions to only a single pre-existing group, dramatically reducing the amount of data that must be stored for each file. The downside of this compression is that if a group doesn't exist with the required members, then the owner may have to grant access to a larger group than they would otherwise like to.
5. Implementation is straight forward: there is a `Users` table and a `Permissions` table that act as an ACL linking users to permissions within a space. To add groups, you could first add a new table to indicate which users are members of which groups.
6. When the user authenticates, you can then look up the groups that user is a member of and add them as an additional request attribute that can be viewed by other processes. *Listing 8.1* shows how groups could be looked up in the `authenticate()` method in the `UserController` after the user has successfully authenticated.
7. When building dynamic SQL queries, be sure to use only placeholders and never include user input directly in the query being built to avoid SQL injection attacks, which are discussed in chapter 2. Some databases support temporary tables, which allow you to insert dynamic values into the temporary table and then perform an SQL `JOIN` against the temporary table in your query. Each transaction sees its own copy of the temporary table, avoiding the need to generate dynamic queries.
8. Drawback: consider what would happen if you changed your API to use an external user store such as LDAP or an OpenID Connect identity provider. In these cases, the groups that a user is a member of are likely to be returned as additional attributes during authentication (such as in the ID token JWT) rather than exists in the API's own database.

#### 8.1.1 LDAP Groups

9. Users are managed centrally in an LDAP (Lightweight Directory Access Protocol) directory. LDAP is designed for storing user information and has built-in support for groups.
10. The LDAP standard defines the following two forms of groups:
    - Static groups are defined as using (A) the `groupOfNames` or (B) `groupOfUniqueNames` object classes, which explicitly list the members of the group using the `member` or `uniqueMember` attributes. The difference between the two ist that `groupOfUniqueNames` forbids the same member being listed twice.
    - Dynamic groups are defined using the `groupOfURLs` object class, where the membership of the group is given by a collection of LDAP URLs that define search queries against the directory. Any entry that matches one of the search URLs is a member of the group.
11. Some of the directory servers also support virtual static groups, which look like static groups but query a dynamic group to determine the membership.
12. To find which static groups a user is a member of in LDAP, you must perform a search against the directory for all groups that have that user's distinguished name as a value of their `member` attribute.
13. You need to connect to the LDAP server using the Java Naming and Directory Interface (JNDI) or another LDAP client library. Normal LDAP users typically are not permitted to run searches, so you should use a separate JNDI `InitialDirContext` for looking up a user's groups
14. *Common types of LDAP groups*
    - Static groups, dynamic groups, virtual static group. Static and dynamic groups are standard, and virtual static groups are nonstandard but widely implemented.
    - *Given the following LDAP filter*: `(&(objectClass=#A)(member=uid=alice,dc=example,dc=com))` which one of the following object classes would be inserted into the position marked `#A` to search for static groups Alice belongs to **`groupOfNames` (or `groupOfUniqueNames`)**.

### 8.2 Role-based access control as in DB

15. Group permissions disadvantage:
  1. to work out who has access to what, you still often need to examine the permissions for all users as well as the groups they belong to
  2. No matter how you organize users into groups you would want another group
    - example: the LDAP directory might just have a group for all software engineers, but your API needs to distinguished between back-end and front-end engineers, QA, and scrum masters.
  3. Finally, even when groups are a good fit for an API there may be large numbers of fine-grained permissions assigned to each group, making it difficult to review the permissions.
16. Role-based access control (RBAC) introduces the notion of role as an intermediary between users and permissions; permissions are **assign to roles, not to invdividuals**.
17. Differences between RBAC and groups
  1. Groups are used primarily to organize users, while roles are mainly used as a way to organize permissions.
  2. Groups tend to be assigned centrally, whereas roles tend to be specific to a particular application or API. As an example, every API may have an admin role, but the set of users that are administrators may differ from API to API
  3. Group-based systems often allow permission to be assigned to individual users, but RBAC systems typically don't allow that. This restriction can dramatically simplify the process of reviewing who has access to what.
  4. RBAC systems split the definition and assigning of permissions to roles from the assignment of users to those roles. It is much less error-prone to assign a user to a role than to work out which permissions each role should have
18. To map role permissions, Java EE uses the `@` notation or retain an explicit notion of lower-level permissions
    - **We have that in C# in the `msconfig`'s `authorization` section**
19. To map users to the roles from DB a table
20. Security domain (realm): roles/groups specific to area in the app
21. Determine which role a user has when they make a request to the API and the permissions that each role allows (see 8.2.3), that is a static assignment.   
22. Dynamic assignment (see 8.2.4) allows more dynamic queries to determine which roles a user should have (they add more constraints to reduce risks of misuse).
    - example: deny the role when out of shift
23. Pre-defined roles that carry a specific set of privileges associated with them and to which subjects they are assigned

### 8.3 Attribute-based access control

24. Attributes
    - About subject = `user`
    - About resource = `object accessed`
    - About action = `what to do`
    - About environment = `context`
25. Access rights are granted to users through the use of policies which combine attributes together. The policies can use any type of attributes. This model supports `boolean` logic, in which rules contain `If...Then` statements about who is making the request, for which resource, and what action.
    - Example: `IF` the requester is a manager, `THEN` allow read/write access to sensitive data
    - Called **policy-based access control** or **claims-based access control**
26. Typically access control decisions are structured as a set of independent rules by describing whether a request should be permitted or denied (see 8.3.1)
27. The safest option is to default to denying requests unless explicitly permitted by some access rule, and to give deny decisions priority over permit decisions. This requires at least one rule to match and decide to permit that action and no rules to decide to deny the action for the request to be allowed.
28. Policy is expressed in the form of rules or domain-specific language (DSL) explicitly designed dto express access control decisions.
29. "Drools" can be used to write all kinds of business rules and provide a convenient syntax for authoring access control rules.
30. ABAC enforcement can be complex as policies increase in complexity (see 8.3.3)
31. Rather than combining all the logic of enforcing policies into the agent itself, another approach is to centralize the definition of policies in a separate server, which provides a REST API for policy agents to connect to and evaluate policy decisions
32. **XACML** is the "eXtensible Access-Control Markup Language", a standard produced by the OASIS standards body. XACML defines a rich, XML-based policy language and a reference architecture for distributed policy enforcement (can be installed).
33. **Best practices for ABAC (attribute-based access control)**
  1. Layer ABAC over a simpler access control technology such as RBAC. This provides a defense-in-depth strategy so that a mistake in the ABAC rules doesn't result in a total loss of security.
  2. Implement automated testing of your API endpoints so that you are alerted quickly if a policy change results in access being granted to unintended parties.
  3. Ensure access control policies are maintained in a version control system so that they can be easily rolled back if necessary. Ensure proper review of all policy changes.
  4. Consider which aspects of policy should be centralized and which should be left up to individual APIs or local policy agents. Though it can be tempting to centralize everything, this can introduce a layer of bureaucracy that can make it harder to make changes. In the worst case, this can violate the principle of least privilege because overly broad policies are left in place due to overhead of changing them.
  5. Measure the performance overhead of ABAC policy evaluation early and often
34. ABAC decisions can be centralized using a policy engine. The XACML standard defines a common model for ABAC architecture, with separate components for policy decisions (PDP), policy information (PIP), policy administration (PAP), and policy enforcement (PEP).

## Michael

In [chapter 03](/chapter03.md) we implemented a simple Access Control List (ACL), with a table of `Permissions` granting a particular user (by username) access to a particular space (by space ID). The allowed access was implemented using a three character string, with `r` giving read privileges, `w` giving write privileges, and `d` giving delete privileges. This is fine initially, but if you were to track each user's access to each space, this number would grow exponentially; in the worst case, every user would need a permission for every space, and every space would need a permission for every user. Given 100 users and 100 spaces, this is 10,000 permissions. Given 1,000,000 users and 1,000,000 spaces, that ramps up to 1,000,000,000,000 permissions, or rows in the database.

Not only is that costly to store and access, if permissions aren't regularly removed as they are no longer needed, users continue to accumulate permissions they don't use and violate the *principle of least privilege*.

To prevent these issues, this chapter covers non-identity-based access control models including Role-based Access Control (RBAC), Groups, and Attribute-based Access Control (ABAC).

## Group-based access control (Section 8.1)

Instead of assigning each individual user a permission, it is often more efficient to assign each user a "group" and to give permissions to the group. A user within a group that has that permission will inherit it directly from the group (e.g. if a group has access to a space, the user will have access to a space).
A group can have many users and a user can have many groups, but groups can also be members of other groups (sub-groups), allowing great extensibility (e.g. a group of employees can also contain a group of managers).

Not only does this method help to reduce the rows in the database, it also ensures that users in a group all have consistent permissions. If every manager should be able to create tickets and assign tasks, then adding a new manager to the group is simpler than adding both the permissions, which also helps prevents accidentally forgetting to grant a permission as well.

In the .NET implementation, we accomplished this by adding a `Group` model to Entity Framework Core, adding a collection of `Group` to the `User` model as well.

```c#
// Group.cs
public class Group
{
  [Key]
  public string GroupId { get; private set; }
  
  public ICollection<User> Users { get; private set; } = new List<User>();

  public Group(string groupId)
  {
    GroupId = groupId;
  }
}

// User.cs
public class User
{
  [Key]
  public string Username { get; private set; }
  [JsonIgnore]
  public string PasswordHash { get; private set; }
  [JsonIgnore]
  public ICollection<Group> Groups { get; private set; } = new List<Group>();

  public User(string username, string passwordHash)
  {
    Username = username;
    PasswordHash = passwordHash;
  }
}
```

Then, in the `AuditMiddleware.cs` class, we can get the logged in user's groups and apply them to the `HttpContext` via `HttpContext.Items["groups"] = groups`.

```c#
private static ICollection<string> GetGroups(string username, NatterDbContext dbContext)
{
  return dbContext.Users
    .Find(username)
    .Groups
    .Select(g => g.GroupId)
    .ToList();
}
```

And finally we can update the `Permissions` object to explicitly represent that the `Username` field could also be a `Groupname` by renaming `Username -> UsernameOrGroupname`.

> **EF Core Tip**
>
> Once we enabled [nullable reference types](https://docs.microsoft.com/en-us/dotnet/csharp/nullable-references) it was rather annoying to see all the warnings for an entity, suggesting we define a default value for each of them. After a bit of research, I solidified some EF concepts in my head that I thought would be useful to share.
>
> For starters, EF has access to private fields; with `private` setters on all properties, EF will use the default constructor to create an instance of the class, then use the private setters to populate the properties. This allows for better restriction, helping prevent misuse of the models. But even further than that, EF has the ability to [define a constructor with parameters matching the property names (allowing for camel-casing)](https://docs.microsoft.com/en-us/ef/core/modeling/constructors). EF will find the best constructor, and populate the matching properties using the parameters. And this doesn't have to be a full constructor either, some properties can remain `private`, allowing you to mix and match the constructor and `private` assignments, however best meets the needs of your model.
>
> It is worth noting that you cannot pass navigational properties in the constructor, as in `User.Groups` above, and they will need to be set using private setters.
>
> With this in mind, you can rectify a lot of nullable reference type complaints this way.
>
> One final tip from the EF docs is a method of making an ID not assignable by the user, but still allow EF Core to automatically generate it. [See the "Read-only properties" section of the docs](https://docs.microsoft.com/en-us/ef/core/modeling/constructors#read-only-properties) for this really cool pattern.

Finally, to wrap things up, inside the `AuthFilterAttributes.cs` class, we can update the `GetPermissions` class to instead return a collection of permissions, and then search that for the one we need. EF Core makes this search trivial.

```c#
private static ICollection<string> GetPermissions(
  HttpContext context,
  int spaceId,
  string username,
  ICollection<string> groups
)
{
  var dbContext = context.RequestServices.GetRequiredService<NatterDbContext>();

  return dbContext.Permissions
      .Where(p => p.SpaceId == spaceId && (
          p.UsernameOrGroupname == username || groups.Contains(p.UsernameOrGroupname)
      ))
      .Select(p => p.Permissions)
      .ToList();
}
```

Madden points out the fact that we are doing two queries here where one would be sufficient

1. Once a user logs in, look up their groups
2. When checking permissions, get a list of group and user-specific permissions

This could be concatenated into a single lookup that is more efficient, however that mixes the authentication and access control layers; whenever possible, you should get the user authenticated in one step, then figure out if their attributes allow access in the separate step.
Importantly, user attributes are not always stored in the same spot, and while the double lookup would work for a database with both the identity and attributes in the same spot, it wouldn't work for something like OpenID Connectivity where attributes could be found on the ID token JWT.

#### LDAP groups (Section 8.1.1)

LDAP, or [Lightweight Directory Access Protocol](https://ldap.com/basic-ldap-concepts/), is used for tracking user information and has built-in support for groups using two (or three) methods

1. Static groups - explicitly list group members
2. Dynamic groups - provide a set of search queries where any matching users are considered part of the group
3. Virtual static groups - look like static groups but behave more like dynamic groups
