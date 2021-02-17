import { useState, useRef } from "react";
import { Button, Input, Label } from "reactstrap";

const FuncRef = () => {
    const inputRef = useRef(null);
    const [ job, setJob ] = useState();

    const handleSubmit = (e) => {
        e.preventDefault();

        setJob(inputRef.current.value); // notice "current" is required
    }

    return (
        <section>
            <h2>Functional Ref Example</h2>
            <form onSubmit={handleSubmit}>
                <Label>Enter your job title:</Label>
                <Input ref={inputRef} placeholder="Software Engineer" />
                <Button color="primary" type="submit">Submit</Button>
            </form>
            { job && <p>You are a { job }? Sounds fun.</p>}
        </section>
    )
}

export default FuncRef;
