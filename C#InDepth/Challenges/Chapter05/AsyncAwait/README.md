# Chapter 05 Challenge - Async/Await

In this challenge, you have access to a generic Fetch service that accepts either a `todo` or `user` type as a type argument into a `GetAll<T>` method. This method returns a `Task` resulting in a `List<T>`.

Utilizing the `await` keyword and the ability to run tasks in parallel, begin retrieving both a list of `users` and `todos`, then find the username of the user with the most incomplete todos.

The answer can be found within the `answer.txt` file, but that isn't too important. What's more important is how and when to use the `await` keyword to optimize your calls to the API. Since you can't build up an expression, there's no point in delaying the request for data; you should get it as soon as you can, and in parallel.

You can score bonus points in this challenge if you properly configure your `Main` method entry point to be asynchronous as well.

## Potential Book Topics Utilized

- Generics (generic fetch service)
- Async/await
- Async entry points
- LINQ
