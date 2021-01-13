import { Component } from "react";

export default class Stateful extends Component {
    constructor(props) {
        super(props);

        this.state = {
            counter: 0
        };

        this.increment = this.increment.bind(this);
    }

    increment() {
        this.setState(prev => {
            return {
                ...prev,
                counter: prev.counter + 1
            }
        });
    }

    render() {
        return (
            <div className="stateful">
                {this.state.counter}

                <button onClick={this.increment}>+</button>
            </div>
        )
    }
}
