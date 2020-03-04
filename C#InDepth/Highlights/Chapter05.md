# Chapter 05 - Writing Asynchronous Code

- The Windows Runtime Platform is commonly known as *WinRT* (pg 151, para 5, NOTE)

## 5.1 - Introducing asynchronous functions

- ... you don't need to dispose of tasks in general. (pg 153, para 2, NOTE)
- `HttpClient` is in some senses the new and improved `WebClient`... it contains only ashnchronous operations. (pg 154, para 1, NOTE)
- [A] method returns as soon as it hits the await expression. (pg 154, para 8)
- A *continuation* is effectively a callback to be executed when an asyncrhonous operation (or any `Task`) has completed. (pg 155, para 2, DEFINITION)
- The `Task` class has a method specifically for attaching continuations: `Task.ContinueWith` (pg 155, para 2, DEFINITION)

## 5.2 - Thinking about asynchrony

- Essentially, all that `await` in C# does is ask the compiler to build a continuation for you. (pg 156, para 3)
- For more information on `SynchronizationContext`, read Stephen Cleary's MSDN magazine article on the topic (http://mng.bz/5cDw). In particular, pay careful attention if you're an ASP.NET developer; the ASP.NET context can easily trap unwary developers into creating deadlocks within code that looks fine. (pg 157, para 5)

## 5.3 - Async method declarations

- ... the language designers didn't need to include [the `async` keyword] at all. (pg 160, para 3)
- ... the `async` modifier has no representation in the generated code... (pg 160, para 4)
- `Task<TResult>` represents an operation that returns a value of type `TResult`, whereas `Task` need not produce a result at all. (pg 161, para 4)
- The ability to return `void` from an async method is designed for compatibility with event handlers (pg 161, para 6)
- Event subsription is pretty much the only time I'd recommend returning `void` from an asynchronous method. (pg 162, para 4, WARNING)
- ... async methods can be generic, static or nonstatic, and specify any of the regular access modifiers (pg 162, para 5)
- None of the parameters in an async method can use the `out` or `ref` modifiers. (pg 162, para 6)
- ... pointer types can't be used as async method parameter types. (pg 162, para 6)

## 5.4 - Await Expressions

- You also can't use the `await` operator within a lock. (pg 166, para 1)
- ... `lock` statements and asynchrony don't go well together. (pg 166, para 2)

## 5.5 - Wrapping of return values

- ... if the `return` statement occurs within the scope of a `try` block that has an associated `finally` block... the expression used to compute the return value is evaluated immediately... If the `finally` block throws an exception... the whole thing will fail. (pg 167, para 5)

## 5.6 - Asynchronous method flow

- If the operation failed and it captured an exception to represent that failure, the exception is thrown. (pg 169, para 6)
- ... it's up to you to make sure that you always write async methods so they return quickly. (pg 172, para 3)
- ... you should generally avoid performing long-running blocking work in an async method. Separate it out into another method that you can create a `Task` for. (pg 172, para 3)
- When you await an asynchronous operation that's failed, it may have failed a long time ago on a completely different thread. (pg 174, para 3)
- When you await a task, if it's either faulted or canceled, an eception will be thrown... the first exception *within* the `AggregateException` is thrown. (pg 175, para 1)
- Typically, if you want to check the exceptions in detail, the simplest approach is to use `Task.Exception` for each of the original tasks. (pg 175, para 1)
- If you validate the parameters as you would in a normal synchronous code, the caller won't have any indication of the problem until the task is awaited. (pg 177, para 4)
- The idea is to write a *nonasync* method that returns a task and is implemented by validating the arguments and then calling a separate async funciton that assumets the argument has been validated... In C# 7 and above, you can use a local async method. (pg 178, para 4)
- ... create a `CancellationTokenSource` and then ask it for a `CancellationToken`, which is passed to an asynchronous operation... you can pass out the same token to multiple operations and not worry about the interfering with each other. (pg 179, para 3)
- ... if an asynchronous method throws an `OperationCanceledException`... the returned task will end up with a status of `Canceled`... This outputs `Canceled` rather than the `Faulted`... (pg 179, para 4)
- The `ThrowCancellationException` method [in Listing 5.8] doesn't contain any await expressions, so the whole method runs synchronously... (pg 180, para 3, Off to the races?)
- ... cancellation is propagated in a natural fashion. (pg 180, para 4)

## 5.7 - Asynchronous anonymous functions

- ... you can't use asynchronous anonymous functions to create expression trees. (pg 180, para 7, NOTE)
- ... [in an anonymous async function,] the asynchronous operation doesn't start until the delegate is invoked, and multiple invocations create multiple operations. (pg 181, para 1)

## 5.8 - Custom task types in C# 7

- If an async method uses an `await` expression on something that's incomplete, object allocation is unavoidable... In those cases, `ValueTask<Result>` provides no benefit and can even be a little more expensive. (pg 182, para 7)
- ... the async/await infrastruture caches a task that it can return from any async method declared to return `Task` that completed synchronously and without exception. (pg 184, para 4, NOTE)

## 5.9 - Async Main methods in C# 7.1

- Unlike most async methods, and async entry point can't have a return type of `void` or use a custom task type. (pg 186, para 7)
- Async entry points are handy for writing small tools or exploratory code that uses an async-oriented API such as Roslyn. (pg 187, para 2)

## 5.10 - Usage tips

- ... when you're writing library code... you don't want to come back to the UI thread... (pg 188, para 1)
- [The `ConfigureAwait` method] takes a parameter that determines whether the returned awaitable will capture the context when it's awaited. (pg 188, para 2)
- The result of calling `ConfigureAwait(false)` is that the continuation won't be scheduled against the original synchronization context; it'll execute on a thread-pool thread. (pg 188, para 5)
- ... `ConfigureAwaitChecker.Analyzer` NuGet package... (pg 189, para 1)
- If you need to log all task failures... you might want to use `Task.WhenAll`... (pg 190, para 1)
- ... be aware of the dangers of using the `Task<TResult>.Result` property and `Task.Wait()` methods... This can easily lead to a deadlock. (pg 190, para 4)
- Stephen Toub... blog posts... "Should I expose synchronous wrappers for asynchronous methods?"... "Should I Expose asynchronous wrappers for synchronous methods?"... the answer is no in both cases... (pg 190, para 5)
- Even if you don't have any requirements to allow cancellation, I advise providing the option consitently right from the start, because it's painful to add later. (pg 191, para 1)
- The methods `Task.FromResult`, `Task.FromException`, and `Task.FromCanceled` are useful [for creating an already completed task]... For more flexibility, you can use `TaskCompletionSource<TResult>`. (pg 191, para 6/7)
