import ControlledInput from "./ControlledInput";
import FormikExample from "./FormikExample";
import FormikValidationExample from "./FormikValidationExample";
import NoControl from "./NoControl";
import UncontrolledInput from "./UncontrolledInput";

const Chapter05 = () => [
    <ControlledInput key="controlled" />,
    <UncontrolledInput key="uncontrolled" />,
    <NoControl key="no-control" />,
    <FormikExample key="formik" />,
    <FormikValidationExample key="formik-plus" />
]

export default Chapter05;
