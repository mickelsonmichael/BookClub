import { useState } from "react";

const Stateless = () => {
    // logic here
    const [counter, setCounter] = useState(0);

    const increment = () => setCounter(prev => prev + 1);

    return (
        <div className="stateless">
            {counter}

            <button onClick={increment}>+</button>
        </div>
    );
};

export default Stateless;
