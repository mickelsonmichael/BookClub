import { Component } from "react";

export default class RouteTest extends Component {
    render() {
        return(<div>You're at: {this.props.text}</div>)
    }
}