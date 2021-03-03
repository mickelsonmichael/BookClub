import { useState, useRef } from "react";
import { Button, Input, Label } from "reactstrap";

const FuncRef = () => {
    const inputRef = useRef(null);
    const [ job, setJob ] = useState();

    const handleSubmit = (e) => {
        e.preventDefault();

        setJob(inputRef.current.value); // notice "current" is required
    }

    // use innerRef instead of ref for reactstrap wrapping
    // reactstrap will pass the innerRef into the ref of the input component
    return (
        <section>
            <h2>Functional Ref Example</h2>
            <form onSubmit={handleSubmit}>
                <Label>Enter your job title:</Label>
                <Input innerRef={inputRef} placeholder="Software Engineer" />
                <Button color="primary" type="submit">Submit</Button>
            </form>
            { job && <p>You are a { job }? Sounds fun.</p>}
        </section>
    )
}

export default FuncRef;
