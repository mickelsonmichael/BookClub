import React, { Component } from "react";
import { Button, Input } from "reactstrap";

export default class Counter extends Component {
    constructor(props) {
        super(props);

        this.decrement = this.decrement.bind(this);
        this.increment = this.increment.bind(this);

        this.state = {
            count: 0
        };
    }

    render() {
        return(
            <div className="d-flex align-items-center m-4">
                {this.getDecrementButton()}
                <Input
                    disabled
                    value={this.state.count}
                    className="d-inline-block mx-2 w-25 text-center"
                />
                {this.getIncrementButton()}
            </div>
        )
    }

    getDecrementButton() {
        return (
            <Button
                className="decrement"
                color="danger"
                onClick={this.decrement}
            >
                -
            </Button>
        )
    }

    decrement() {
        this.setState(prev => ({
            count: prev.count - 1
        }));
    }

    getIncrementButton() {
        return (
            <Button className="increment" color="success" onClick={this.increment}>+</Button>
        )
    }

    increment() {
        this.setState(prev => ({
            count: prev.count + 1
        }));
    }
}