import { Component } from "react";
import { Formik, Form, Field, ErrorMessage } from "formik";

const initialValues = {
    color: "red"
}

export default class FormikExample extends Component {
    constructor(props) {
        super(props);

        this.handleSubmit = this.handleSubmit.bind(this);

        this.state = {
            squareColor: initialValues.color
        };
    }

    handleSubmit(values) {
        this.setState({
            squareColor: values.color
        })
    }

    render() {
        return(
            <section>
                <h2>Using Formik Example</h2>

                <Formik initialValues={initialValues} onSubmit={this.handleSubmit}>
                    {() => {
                        return (
                        <Form>
                            <Field name="color" />

                            <button type="submit">Change the Color</button>
                        </Form>
                        );
                    }}
                </Formik>

                <span>
                    <div style={{ height: "100px", width: "100px", backgroundColor: this.state.squareColor }}></div>
                </span>
            </section>
        )
    }
}