import { Component, createElement, Children } from "react";
import enroute from "enroute";
import invariant from "invariant";

export default class Router extends Component {
    constructor(props) {
        super(props);

        this.routes = {};

        this.addRoutes(props.children);
        
        this.router = enroute(this.routes);
    }

    addRoute(element, parent) {
        const { component, path, children, ...rest } = element.props;

        invariant(component, `Route ${path} requires a component`);
        invariant(typeof path === "string", `Route ${path} is not a string`);

        const render = (params, renderProps) => {
            const finalProps = {
                ...params,
                ...this.props,
                ...renderProps,
                ...rest
            }
            
            const children = createElement(component, finalProps);

            return parent ? parent.render(params, { children }) : children;
        }

        const route = this.normalizeRoute(path, parent);

        if (children) {
            this.addRoutes(children, { route, render });
        }

        this.routes[this.cleanPath(route)] = render;
    }

    addRoutes(routes, parent) {
        Children.forEach(routes, route => this.addRoute(route, parent));
    }

    cleanPath(path) {
        return path.replace(/\/\//g, '/');
    }

    normalizeRoute(path, parent) {

        if (path[0] === '/') {
            return path;
        }

        if (!parent) {
            return path;
        }

        return `${parent.route}/${path}`;
    }

    render() {
        const { location } = this.props;

        invariant(location, "<Router /> needs a location to work");

        return this.router(location);
    }
}