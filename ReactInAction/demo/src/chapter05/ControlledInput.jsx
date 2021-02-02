import { Component } from "react";

export default class ControlledInput extends Component {
    constructor(props) {
        super(props);

        this.handleOnChange = this.handleOnChange.bind(this);

        this.state = {
            name: "Randy Marsh"
        };
    }

    handleOnChange(e) {
        this.setState({
            name: e.target.value
        })
    }

    render() {
        return <section>
            <h2>Controlled Input</h2>

            <p>
                You control the current state and value of the component. React doesn't manage anything for you.
            </p>

            <label>Enter your name:</label>
            <input type="text" value={this.state.name} onChange={this.handleOnChange} />

            <p>Good afternoon, <strong>{this.state.name}</strong></p>

            <p>
                Clicking this button will change the value of the input: 
                <button onClick={() => this.setState({ name: "Spartacus"})}>I am Spartacus.</button>
            </p>
        </section>
    }
}
