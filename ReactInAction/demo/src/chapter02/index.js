// Instead of import statements, we use a CDN and a <script> tag
// Import statements require Babel and/or Webpack, so to avoid using those
// we can use the CDN directly

//import React from "react";
//import ReactDOM from "react-dom";

const root = document.getElementById("root");

const component = React.createElement("pre", {}, "Hello, world!");

ReactDOM.render(component, root);
