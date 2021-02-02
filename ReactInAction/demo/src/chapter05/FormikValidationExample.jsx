import { Component } from "react";
import { Formik, Form, Field, ErrorMessage } from "formik";
import * as yup from "yup";

const initialValues = {
    color: "red",
    phone: "",
}

const validationSchema = yup.object({
    color: yup.string()
        .required()
        .oneOf([
            "blue",
            "yellow",
            "green",
            "red"
        ]),
    phone: yup.string()
        .matches(/\d{3}-\d{4}/)
})

export default class FormikValidationExample extends Component {
    constructor(props) {
        super(props);

        this.handleSubmit = this.handleSubmit.bind(this);

        this.state = {
            circleColor: initialValues.color
        };
    }

    handleSubmit(values) {
        this.setState({
            circleColor: values.color
        })
    }

    render() {
        return (
            <section>
                <h2>Using Formik with Yup Example</h2>

                <p>
                    Formik allows you to provide your own validation schema, or utilize <a href="https://github.com/jquense/yup" target="_blank" rel="noreferrer">Yup</a>.
                    You can then pass that schema into your Formik form and it will automatically display "ErrorMessage" components whenever necessary, and hide them
                    when the validation is OK. It makes form validation incredibly flexible and simple.
                </p>

                <hr />

                <Formik initialValues={initialValues} onSubmit={this.handleSubmit} validationSchema={validationSchema}>
                    {() => {
                        return (
                            <Form>
                                <label>Select a color for the circle</label>
                                <Field name="color" />
                                <button type="submit">Change the Color</button>

                                <ErrorMessage component="p" style={{ color: "red" }} name="color" />
                            </Form>
                        );
                    }}
                </Formik>

                <span>
                    <div style={{ height: "100px", width: "100px", borderRadius: "100px", backgroundColor: this.state.circleColor }}></div>
                </span>
            </section>
        )
    }
}