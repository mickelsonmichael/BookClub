import React from "react";
import { Button } from "reactstrap";

const Form = () => {
  const [isClicked, setIsClicked] = React.useState(false);

  return (
    <div>
      <Button type="text" onClick={() => setIsClicked(true)}>
        Bootstrap
      </Button>
      {isClicked ? "clicked" : "not"}
      <button
        type="button"
        onClick={() => {
          throw new Error("here it is");
        }}
      >
        Normal
      </button>
    </div>
  );
};

export default Form;
