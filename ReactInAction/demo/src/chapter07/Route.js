import { Component } from "react";
import invariant from "invariant";

export default class Route extends Component {
    render() {
        return invariant(false, "<Route /> components are for config only and shouldn't be rendered");
    }
}