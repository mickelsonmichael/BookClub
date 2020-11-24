# Chapter 08 - Moving Features

As the title suggests, this chapter focuses on refactorings that move code from one place to another, and the destination doesn't necessarily have to be a happy place (see "Remove Dead Code").

## Moving Function

Starting the chapter off simple, moving function is simply moving a function from one class, or context, to another. This is usually motivated because a function is more strongly related to an alternative class than the one it currently resides. It could also be a part of encapsulating certain actions more or even less. If something complex could be wrapped up behind encapsulation that's great, and if something simple could be made easier to use by unwrapping it, that's also great.

## Move Field

**ALERT:** There has been a domain-driven design reference during this chapter. Fowler speaks briefly about how a well modeled domain can lead to easier to manage code, and _Move Field_ is a pathway to getting to the "ideal" model for the domain of interest. There are several reasons he lists that will alert you that you may have a good candidate for _Move Field_.

- Two properties are often passed into functions together
- A property exists in multiple classes at once, so an update to the value requires multiple updates to multiple objects
- A change to the model of that property requires multiple changes instead of one consolidated change

## Move Statements into Function

Whenever a statement could be better understood within the context of a function, whether that's an existing function or a new one, that statement can easily be lifted and placed into the function itself. If you find that it doesn't fit, the reversal is equally trivial.

## Move Statements to Callers

The inverse of _Move Statements into Function_, this refactoring is usually performed whenever the code within the function needs to vary slightly from the rest of the related code. Along the same lines as the example Fowler provides, it could be that code that generates HTML markup needs to do something unique when considering how to output a particular segment. While that may have been fine when the segment was similar to all the other segments, once it is varied or complex enough it may be better suited being moved outside of the function.

## Replace Inline Code with Function Call

Fowler spends most of this very brief (1 page) chapter talking about the motivation behind replacing a bit of inline code with a call to a function instead. Essentially all boiling down to concepts of "clean code" and having verbose function names making your application easier to reason about, as well as making it easier to change and more DRY.

Worth noting, is that this particular refactoring doesn't require you to create the function call yourself, you could instead utilize a library. Doing so not only reduces the amount of code you have to maintain, but also provides a potentially ubiquitous way to perform an action. For example, replacing a loop with a Linq expression instead is easier to read and most C# developers will already know how the Linq function behaves.

## Slide Statements

Perhaps the easiest refactoring that can have the most destructive consequences, _Slide Statements_ simply entails "sliding" a statement up or down a function to be nearer to related data. Fowler references the habit some developers have of declaring all their variables at the top of their functions instead of near where they are used.

I mentioned this was destructive, because if your code base isn't well written, there could be unintended side-effects to some method calls that act on the expression you're moving, which may not be immediately apparent. So here's hoping you have well written tests ;)

## Split Loop

Perhaps one of the more controversial refactorings, _Split Loop_ is just as it sounds, splitting a loop that's doing two separate tasks into two separate loops. As Fowler puts it, "...if you're doing two separate things in the same loop, then whenever you need to modify the loop you have to understand both things."

I don't know that I'm in love with this refactoring. He mentions how you can easily do this for clarity, then, if the multiple loops is a bottleneck, undo it for the sake of speed. But that requires you to do the optimization portion of the programming journey; you now not only need to spend time refactoring, but also spending time optimizing, and in the "quick turnaround" lifestyle that most management teams pressure developers into, that seems like a tall order to me.

## Replace Loop with Pipeline

Translated into C# jargon, this would be called "replace loop with Linq". Linq is often optimized, easier to read, and more well understood by more programmers than any loop could be. No need to reinvent the wheel.

Another bonus of Linq is that thanks to PLinq, it would allow your code to become parallelized very easily without having to make major modifications, all it would require is a call to `AsParallel()` and you're off to the races.

## Removing Dead Code

This one is possibly the hardest for me to actually pull the trigger on at times. Simple enough, _Removing Dead Code_ entails removing code that is no longer referenced or used anywhere else in the application. With Git (or other modern source control solutions), you can always return to the commit history to retrieve a function you needed, so there's no real reason to keep the unreferenced code around, even if you "might need it later."
 
