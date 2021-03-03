import { Component } from "react";
import { Button, Input, Label } from "reactstrap";

export default class ClassRef extends Component {
    constructor(props) {
        super(props);

        this.handleFormSubmit = this.handleFormSubmit.bind(this);

        this.input = {};
        this.state = {
            result: ""
        }
    }

    handleFormSubmit(e) {
        e.preventDefault();

        this.setState({
            result: this.input.value
        })
    }

    // use innerRef instead of ref for reactstrap wrapping
    // reactstrap will pass the innerRef into the ref of the input component
    render() {
        return (
            <section>
                <h2>Class Ref Example</h2>
                <form onSubmit={this.handleFormSubmit}>
                    <Label>Enter your name</Label>
                    <Input innerRef={(node) => { this.input = node }} placeholder="John Doe" />
                    <Button color="primary" type="submit">Submit</Button>
                </form>
                { this.state.result && <p>Hi, {this.state.result}</p> }
            </section>
        )
    }
}