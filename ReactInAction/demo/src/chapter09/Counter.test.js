import React from "react";
import { shallow } from "enzyme";
import renderer from "react-test-renderer";

import Counter from "./Counter";

describe("<Counter />", () => {
    let shallowCounter;

    beforeEach(() => {
        shallowCounter = shallow(<Counter />);
    })

    test("snapshot", () => {
        const component = renderer.create(<Counter />);

        const tree = component.toJSON();

        expect(tree).toMatchSnapshot();
    })

    it("starts the count at zero", () => {
        expect(shallowCounter.state().count).toBe(0);
    })

    it("increments count when + button clicked", () => {
        shallowCounter.find(".increment").simulate("click");

        expect(shallowCounter.state().count).toBe(1);
    })

    it("decrement count when - button clicked", () => {
        shallowCounter.find(".decrement").simulate("click");

        expect(shallowCounter.state().count).toBe(-1);
    })
});
