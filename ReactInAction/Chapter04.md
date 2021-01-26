# Chapter 04

## Mike

One particular item of note is the fact that React is changing the "will" lifecycle methods. See <https://reactjs.org/blog/2018/03/27/update-on-async-rendering.html> for the full blog post.

Essentially, the React team felt that the "will" lifecycle methods

- `componentWillMount`
- `componentWillReceiveProps`
- `componentWillUpdate`

These methods were considered too easy to abuse and lead to many developers using unsafe coding practices. They also were not ideal for asynchronous rendering of components. So these methods are being deprecated. As they were slowly removed, they became less and less accessible.

Initially, in React 16.3, they were given an alias with an `UNSAFE_` prefix (i.e. `UNSAFE_componentWillMount`) to help guide the developer away from using a feature that will eventually be removed from the API. Then a deprecation error was added in a later v16.x release. Now, in the latest versions of react (>17.0), the properties are completely removed and *only* the `UNSAFE_` versions are available.

But the React team isn't just removing lifecycle methods, they are replacing them with some **new** lifecycle methods.

### `getDerivedStateFromProps(props, state)`

A static method that is "invoked after a component is instantiated as well as before it is re-rendered." This will return the modified state (or `null` for no modifications). This is helpful when your state needs to be modified based on the props being passed to your component.

### `getSnapshotBeforeUpdate(prevProps, prevState)`

This method is "called right before mutations are made (e.g. before the DOM is updated." It can be considered a partial replacement for `componentWillUpdate`. The value returned from the method is passed as a third parameter to the `componentDidUpdate` method.

