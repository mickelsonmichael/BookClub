# Chapter 06 - Async implementation

- "... in debug builds, the generated state machines are classes rather than structs."
  - pg 194, para 4, *Debug and release builds differ, and future implementations may, too*
 - “The differences mostly affect performance” pg 194 para 6, Debug and release builds differ, and future implementations may, too*
- "I'm going to present only asynchronous methods, not async anonymous functions; the machinery between the two is the same anyway"
  - pg 195, para 1

## 6.1 Structure of the Generated Code

- "... the implementation... is in the form of a *state machine*. The compiler will generate a private nested struct to represent the asynchronous method"
  - pg. 195, para. 2
 -  “I’m going to talk about the state machine pausing. This corresponds to a point where the async method reaches an await expression and the operation being awaited hasn’t completed yet”  pg 195 para 3, Note*
 - “four kinds of state, in common execution order: Not started, Executing, Paused, Complete (either successfully or faulted)” pg 195 para 4
- “Only the Paused set of states depends on the structure of the async method.” Pg 195 para 5
- “The state is recorded when the state machine needs to pause; the whole purpose is to allow it to continue the code execution later from the point it reached.” Pg 195 para 5
- "I typically use a mixture of ildasm and Redgate Reflector for this sort of work, setting the Optimization level to C# 1 to prevent the decompiler from reconstructing the async method for us."
  - pg. 196, para. 4, *Do try this at home*
- "...`AsyncTestMethodBuilder`... is a value type, and it's part of the common async infrastrucutre."
  - pg. 198, para. 1
- [AsyncTaskMethodBuilder] “The builder provides functionality that the generated code uses to propagate success and failure, handle awaiting, and so forth.” Pg 198 para 4
- "Both the state machine and the `AsyncTaskMethodBuilder` are *mutable* value types."
  - pg. 198, para. 5
- "... mutable value types and public fields are almost always a bad idea."
  - pg. 199, para. 1
- “The state is just an integer with one of the following values: –1—Not started, or currently executing (it doesn’t matter which), –2—Finished (either successfully or faulted),  Anything else—Paused at a particular await expression” pg200 para 2
- “The crucial point to remember is that you need fields only for values that you need to come back to after the state machine resumes at some point.” Pg 200 para 4
- “Only one awaiter is relevant at a time, because any particular state machine can await only one value at a time.” Pg 200 para 5
- "Before C# 7, the builder type was always `AsyncVoidMethodBuilder`, `AsyncTaskMethodBuilder`, or `AsyncTaskMethodBuilder<T>`. With C# 7 and custom task types, the builder type specified by the `AsyncTaskMethodBuilderAttribute` is applied to the custom task type."
  - pg. 200, para. 5
- "... you need fields only for values that you need to come back to after the state machine resumes at some point."
  - pg. 200, para. 6
- "The compler creates a single field for each awaiter type that's used."
  - pg. 200, para. 7
- "If a local variable is used only between two await expressions rather than *across* await expressions, it can stay as a local variable in the `MoveNext()` method."
  - pg. 201, para. 2
- "... stack variables... are introduced when an await expression is used as part of a bigger expression and some intermediate values need to be remembered."
  - pg. 201, para. 5
- "Unlike real local variables, the compiler does reuse temporary stack variables of the same type and generates only as many fields as it needs to."
  - pg. 202, para. 4
- “MoveNext() returns if any of the following occurs: The state machine needs to pause to await an incomplete value. Execution reaches the end of the method or a return statement. An exception is thrown but not caught in the async method. Note that in the final case, the MoveNext() method doesn’t end up throwing an exception. Instead, the task associated with the async call becomes faulted” pg 202 para 7
- "... the `MoveNext()` method... return type is `void`, not a task type."
  - pg. 203, para. 1

## 6.2 A simple `MoveNext()` implementation

- "... the `MoveNext()` method is invoked once when the async method is first called and then once each time it needs to resume from being paused at an await expression."
  - pg. 207, para. 3
- "If every await expresion takes the fast path, `MoveNext()` will be called only once.`
  - pg. 207, para. 3
- "... special exceptions (`ThreadAbordexception` and `StackOverflowException`, for example) will ever cause `MoveNext()` to end with an exception"
  - pg. 208, para. 1
- "...the start of the `MoveNext()` method is always effectively a `switch` statement used to jump right to the piece of code within the method based on the state."
  - pg. 208, para. 2
- "`MoveNext()` should neer end up being called in the executing or completeed states."
  - pg. 208, para. 4, *What about other states?*
- "There aren't... distinct state numbers for not started and executing: both use -1"
  - pg. 208, para. 4, *What about other states?*
- "Within the state machine, `return` is used when the state machine is paused after scheduling a continuation for an awaiter."
  - pg. 208, para. 5
- "You have to call `GetResult()` even if there isn't a resut `value` to let the awaiter propogate errors if necessary."
  - pg. 209, para. 3, Item 9
- "The call to b`builder.AwaitUnsafeOnCompleted(ref awaiter1, ref this)` is the part that does the boxing dance with a call back into `SetStateMachine`... and schedules the continuation."
  - pg. 210, para. 3
- "...even small async methods - even those using `ValueTask<TResult>` - can't be sinsibly inlined by the JIT compiler."
  - pg. 210, para. 6

## 6.3 How control flow affects `MoveNext()`

- "...it's always been valid to use `await` in a `try` block, but in C# 5, it was invalid to use it in a `catch` or `finally` block. That restriction was lifted in C# 6..."
  - pg. 213, para. 5
- "...even in IL, you're not allowed to jump from outside a `try` block to inside it."
  - pg. 215, para. 1
- "If the `finally` block is executing because you're passing the state machine and returning to the caller, the code in the original async method's `finally` block shouldn't execute."
  - pg. 215, para. 4
- "...where I've always used a `switch` statement for "jump to X" pieces of code, the compiler can sometimes use simpler branching code. Consistency... doesn't matter to the complier."
  - pg. 216, para. 1

## 6.4 Execution contexts and flow

- "Execution contexts aren't like [other contexts]; you pretty much always want the same execution context when your async method continues, even if it's on a different thread."
  - pg. 217, para. 1
- "...preservation of the execution contet is called *flow*. An execution context is said to flow across await expressions, meaning that all your code operates in the same execution context."
  - pg. 217, para. 2
- "`ICriticalNotifyCompletion.UnsafeOnCompleted` is marked with `[SecurityCritical]`. It can be called only by trusted code, such as the framework's `AsyncTaskMethodBuilder` class."
  - pg. 217, para. 3s

## 6.5 Custom task types revisited
