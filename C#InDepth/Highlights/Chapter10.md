# Chapter 10 - A smorgasbord of features for concise code
## 10.1 - Using static directives
- [a using static directive] …the following members are available directly by using their simple
names rather than having to qualify them with the type: Static fields and properties, Static methods, Enum values, Nested types
  - pg 286, para. 3
- …the members that are available via the static imports are considered during member lookup only after other members have been considered
  - pg 288, para. 2
- The static part of using static doesn’t mean that the type you import must be static.
  - pg 288, para. 3 **The Imported type doesn’t have to be static**
- Extension methods from a single type can be imported with a using static directive for that type without importing any extension methods from the rest of the namespace.
  - pg 288, para 6 **first bullet point**
- Extension methods imported from a type aren’t available as if you were calling a regular static method like Math.Sin. Instead, you have to call them as if they were instance methods on the extended type.
  - pg 288, para. 6 **second bullet point**
- If you wish to include some extension methods but allow users to explicitly opt into them, I encourage the use of a separate namespace for that purpose.
  - pg 289, para. 3
- Extension methods discovered via static imports aren’t preferred over extension methods discovered through namespace imports.
  - pg 290, **Note**
  
 ## 10.2 - Object and collection initializer enhancements
- ...object initializers can now use indexers, and collection initializers can now use extension methods.
  - pg 290 para. 5
- ...when should you use an indexer in an object initializer rather than a collection initializer? 1. If you can’t use a collection initializer because the type doesn’t implement IEnumerable or doesn’t have suitable Add methods. 2. If the indexer and the Add method would handle duplicate keys in the same way. 3. If you’re genuinely trying to replace elements rather than adding them.
  - pp 292, 293, para. 6 (292)
- I encourage you to think through the pros and cons for yourself; don’t blindly trust my advice or anyone else’s.
  - pg 297 **Note**
- When a method is obscured by explicit interface implementation, that’s often meant to discourage you from calling it without a certain amount of care.
  - pg 298, para. 3
- …object and collection initializers are usually used in two places: Static initializers for collections that’ll never be modified after type initialization, Test code
  - pg 299, para. 1

## 10.3 - The null conditional operator
- …using == (which already handles null correctly, at least for references…
  - pg 300, para. 2
- …the *null conditional ?*. operator, which is a short-circuiting operator that stops if the expression evaluates to null.
  - pg 300, para. 2
- …the null conditional operator can also be used to access methods, fields, and indexers.
  - pg 300, para.5
- If the type of the overall expression would be a non-nullable value type without the null conditional operator, it becomes the nullable equivalent if a null conditional operator is involved anywhere in the sequence.
  - pg 300, para. 5
- Anytime you use the null conditional operator in any kind of condition, you need to consider three possibilities: Every part of the expression is evaluated, and the result is true. Every part of the expression is evaluated, and the result is false. The expression short-circuited because of a null value, and the result is null.
  - pp 301, 302, para. 6 (301)
- …the result of the expression is always classified as a value rather than a variable.
  - pg 305, para. 1
  
## 10.4 - Exception filters

- The basic premise is that you can now write catch blocks that only sometimes catch an exception based whether a filter expression returns true or false. If it returns true, the exception is caught. If it returns false, the catch block is ignored.
  - pg 305, para. 4
- I can see exception filters being useful in two generic use cases: retry and logging.
  - pg 306, para. 2
- …the contextual keyword when followed by an expression in parentheses that can use the exception variable declared in the catch clause and must evaluate to a Boolean value.
  - pg 307, para. 2
- …you shouldn’t use finally for anything security sensitive.
  - pg 310 para. 2 **Security impact of the two-pass model**
- …it’s worth being aware of every layer of your code that might be attempting to retry a failed operation.
  - pg 311, para. 2 **Keep track of your retry policies**
- …at least sometimes, it’s useful to log an exception within one method call even if it’s going to be caught (and possibly logged a second time) somewhere further down the stack.
  - pg 312, para. 5
- I strongly urge you *not* to filter based on the exception message… Code that behaves differently based on a particular exception message is fragile.
  - pg 314, para. 2
- …although a simple throw statement does preserve the original stack trace for the most part, subtle differences can exist, particularly in the stack frame where the exception is caught and rethrown.
  - pg 314, para.4
