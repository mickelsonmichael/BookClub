import { Component } from "react";
import { Formik, Form, Field } from "formik";

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

                <p>
                    <a href="https://formik.org" target="_blank" rel="noreferrer">Formik</a> is a forms library that simplifies management of form state.
                    It is a particularly simple, yet magical library. You can watch the creator make <em>most</em> of the features in one hour on their
                    website; it is that simple a concept.
                </p>

                <p>
                    You simply need to define an object that will serve as an initial state (or just pass it directly into Formik).
                    Formik will then update any "Field" elements with the proper values, and update the backing store with the user changes.
                </p>

                <hr />

                <Formik initialValues={initialValues} onSubmit={this.handleSubmit}>
                    {({ setFieldValue }) => {
                        return (
                        <Form>
                            <label>Select a color for the square: </label>
                            <Field name="color" />
                                <button onClick={() => setFieldValue("color", "red")}>Reset</button>

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