# Chapter 06 - Async implementation

- "... in debug builds, the generated state machines are classes rather than structs."
  - pg 194, para 4, *Debug and release builds differ, and future implementations may, too*
- "I'm going to present only asynchronous methods, not async anonymous functions; the machinery between the two is the same anyway"
  - pg 195, para 1

## 6.1 Structure of the Generated Code

- "... the implementation... is in the form of a *state machine*."
  - pg. 195, para. 2
- "I typically use a mixture of ildasm and Redgate Reflector for this sort of work, setting the Optimization level to C# 1 to prevent the decompiler from reconstructing the async method for us."
  - pg. 196, para. 4, *Do try this at home*
- "...`AsyncTestMethodBuilder`... is a value type, and it's part of the common async infrastrucutre."
  - pg. 198, para. 1
- "Both the state machine and the `AsyncTaskMethodBuilder` are *mutable* value types."
  - pg. 198, para. 5
- "... mutable value types and public fields are almost always a bad idea."
  - pg. 199, para. 1
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
- "... the `MoveNext()` method... return type is `void`, not a task type."
  - pg. 203, para. 1

## 6.2 A simple `MoveNext()` implementation

## 6.3 How control flow affects `MoveNext()`

## 6.4 Execution contexts and flow

## 6.5 Custom task types revisited
