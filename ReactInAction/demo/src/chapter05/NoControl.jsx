import { Component } from "react";

export default class NoControl extends Component {
    constructor(props) {
        super(props);

        this.state = {
            numberOfCats: 2
        };
    }

    render() {
        return (
            <section>
                <h2>No Control</h2>

                <p>
                    You won't be able to update the value without providing a method that listens to the events.
                    React will not update the DOM to reflect the keys you press, or keys you press without it.
                </p>

                <input type="number" value={this.state.numberOfCats} />

                <p>What a coincidence, I also own {this.state.numberOfCats} cat(s).</p>

                <p>
                    You can, however, still update the state and the value will be updated, try clicking the button to force an update

                    <button onClick={() => this.setState(p => ({ numberOfCats: p.numberOfCats - 1 }))}>Less Cats</button>

                    <button onClick={() => this.setState(p => ({ numberOfCats: p.numberOfCats + 1}))}>More Cats</button>
                </p>
            </section>
        )
    }
}