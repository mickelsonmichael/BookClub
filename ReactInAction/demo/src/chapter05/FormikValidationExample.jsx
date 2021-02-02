import { Component } from "react";
import { Formik, Form, Field, ErrorMessage } from "formik";
import * as yup from "yup";

const initialValues = {
    color: "red"
}

const validationSchema = yup.object({
    color: yup.string()
        .required()
        .oneOf([
        "blue",
        "yellow",
        "green",
        "red"
    ])
})

export default class FormikValidationExample extends Component {
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
        return (
            <section>
                <h2>Using Formik with Yup Example</h2>

                <Formik initialValues={initialValues} onSubmit={this.handleSubmit} validationSchema={validationSchema}>
                    {() => {
                        return (
                            <Form>
                                <Field name="color" />
                                <button type="submit">Change the Color</button>

                                <ErrorMessage component="p" style={{ color: "red" }} name="color" />
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