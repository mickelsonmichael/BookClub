import { Component } from "react";

export default class UncontrolledInput extends Component {
    constructor(props) {
        super(props);
        
        this.handleChange = this.handleChange.bind(this);

        this.state = {
            favoriteFood: "sushi" // default will not be given to the input without setting `defaultValue`
        }
    }

    handleChange(e) {
        this.setState({
            favoriteFood: e.target.value
        })
    }

    render() {
        return (
            <section>
                <h2>Uncontrolled Input</h2>

                <p>
                    You allow React/HTML to handle the input, you simply record the changes to that input.
                    Providing `defaultValue` will allow you to define the value when the component is initially rendered,
                    but you will lose "control" after that point.
                </p>

                <label>What is your favorite food?</label>
                <input
                    type="text"
                    defaultValue={this.state.favoriteFood}
                    onChange={this.handleChange}
                />

                <p>I too love {this.state.favoriteFood}</p>

                <p>
                    Clicking the button and setting the state will have no impact on the input itself,
                    but will still update the store, leading to inconsistencies

                    <button onClick={() => this.setState({ favoriteFood: "chocolate milk"})}>I like chocolate milk.</button>
                </p>
            </section>
        );
    }
}