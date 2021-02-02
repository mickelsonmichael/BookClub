import { Component } from "react";

const initialState = {
    books: ["Pragmatic Programmer", "React in Action"],
    name: "Wanda",
    job: {
        description: "programmer",
        salary: 1000000,
        benefits: {
            retirement: "401k",
            health: "HSA"
        }
    },
    degree: {
        school: "Michigan State",
        gpa: 3.0
    }
};

export default class ShallowMerge extends Component {
    constructor(props) {
        super(props);

        this.state = initialState;

        this.handleButtonClick = this.handleButtonClick.bind(this);
        this.handleReset = this.handleReset.bind(this);
    }

    handleReset() {
        this.setState(initialState);
    }

    handleButtonClick() {
        // setState only performs a SHALLOW MERGE
        // while most of our information about Wanda is left intact
        // we lose the `benefits` property, because it is too deep for a shallow merge to copy
        this.setState((prev) => {
            return {
                job: {
                    description: "senior programmer",
                    salary: "1200000"
                }
            }
        }, () => console.log("done"));
    }

    render() {
        console.log(this.state);
        
        return (
            <section>
                <p>
                    `setState` only performs a <strong>shallow merge</strong>.
                    When the button is clicked, most of the state is left intact,
                    but we lose the `benefits` information because it is too deep for a shallow merge to copy
                </p>
                <dl>
                    <dt>Name</dt>
                    <dd>{this.state.name}</dd>

                    <dt>Books</dt>
                    <dd>{this.state.books.join(", ")}</dd>

                    <dt>Job Description</dt>
                    <dd>{this.state.job.description}</dd>

                    <dt>Job Salary</dt>
                    <dd>{this.state.job.salary}</dd>

                    <dt>Retirement Plan</dt>
                    <dd>{this.state.job.benefits?.retirement ?? 'undefined'}</dd>

                    <dt>Health Plan</dt>
                    <dd>{this.state.job.benefits?.health ?? 'undefined'}</dd>

                    <dt>Graduated From</dt>
                    <dd>{this.state.degree.school}</dd>
                    
                    <dt>Graduating GPA</dt>
                    <dd>{this.state.degree.gpa}</dd>
                </dl>

                <button onClick={this.handleButtonClick}>Apply for a new job!</button>
                <button type="reset" onClick={this.handleReset}>Reset</button>
            </section>
        )
    }
} 