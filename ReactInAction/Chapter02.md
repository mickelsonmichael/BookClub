# Chapter 2 - `<Hello World />`: Our First Component

## Mike

This chapter was a bit over-inflated; I say that because the code examples took up a good quarter of the chapter, and they didn't necessarily need to. The reason they took up so much space in the chapter is because the author waited until the last section to introduce the concept of JSX. That meant that he had to write out the inidividual calls to `render` and `createElement`, line by line, and used excessive spacing to do so. What could have been 3 lines of JSX ended up being 24 lines. Whether or not it was worth it, I'm not sure. He could have easily started off the chapter with just the methods, and gone in-depth with how Bable/Webpack convert the JSX into those methods. Instead there was just some hand-waving to say "and then Babel changes the JSX" and that was it.


One of the most important concepts this chapter introduces is regarding components. Components have one relationship set, _parent-child_. While components can be siblings (a component next to another component within the same parent), the components **don't know and don't care** about this relationship. Being ajdacent to another component should not affect the component in any way, all that should matter is what component is has within it, and potentially what component is the parent (via the passed in parameters). Something that is particularly important, but may be challenging for newcomers, is the concept of passing information back up to the parent. This is done using functions instead of properties. The parent passes a reference to a function down to the child(ren), and that function is then called by the child when the information needs to be communicated back to the parent.

Another point he mentions is that the `setState` method isn't necessarily immediate. This is something I want to research a bit more. I'm aware that it is recommended to always use the `setState(previousState => doStuff)` methodology when the previous state matters, but I wasn't clear on why this was the case. I thought I knew, but this revelation may increase the urgency at which I use the feature.

I also learned (or potentially re-learned because I forgot?) that you can return an array of elements to the `render` method instead of just one component. I've always disliked that I've had to wrap my component in a `<div>` tag each time, so finding out that an array is an option is a happy revelation. This was just a passing comment he made in the middle of the chapter, and every point after that he seems to maintain that the `render` method _must_ return a _single_ component (even though that isn't true).

```js
render() {
	return [
		<div>left</div>,
		<div>right</div>
	];
}
```

I have tested the above code and it functions beautifully.

