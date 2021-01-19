import { Component } from "react";

export default class ShallowMerge extends Component {
    constructor(props) {
        super(props);

        this.state = {
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
        }

        this.handleButtonClick = this.handleButtonClick.bind(this);
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
            <div>
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
            </div>
        )
    }
} 