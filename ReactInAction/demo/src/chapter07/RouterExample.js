import { Component } from "react";
import { Button } from "reactstrap";
import Router from "./Router";
import Route from "./Route";
import RouteTest from "./RouteTest";

export default class RouterExample extends Component {

    constructor(props) {
        super(props);

        this.setLocation = this.setLocation.bind(this);

        this.state = {
            location: "route1"
        }
    }

    setLocation(loc) {
        this.setState({ location: loc });
    }

    render() {
        return (
            <section>
                <h2>Book Router</h2>
                <Router location={this.state.location}>
                    <Route path="route1" component={RouteTest} text="route 1" />
                    <Route path="route2" component={RouteTest} text="route 2" />
                </Router>
                <div>
                    <Button color="primary" onClick={() => this.setLocation("route1")}>Go to Route 1</Button>
                    <Button color="success" onClick={() => this.setLocation("route2")}>Go to Route 2</Button>
                </div>
            </section>
        )
    }
}