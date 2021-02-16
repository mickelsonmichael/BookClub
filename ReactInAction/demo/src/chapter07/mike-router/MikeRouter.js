import React, { Component } from "react";

export default class MikeRouter extends Component {
    constructor(props) {
        super(props);

        this.routes = {};
        this.parseRoutes(props.children);
        this.router = this.createRouter();
    }

    parseRoutes(routes, parent) {
        React.Children.forEach(
            routes,
            route => this.parseRoute(route, parent)
        );
    }

    parseRoute(element, parent) {
        const { component, path, children, ...rest } = element.props;

        const render = (params, renderProps) => {
            const finalProps = {
                ...params,
                ...this.props,
                ...renderProps,
                ...rest
            };

            const children = React.createElement(component, finalProps);

            return parent ? parent.render(params, { children }) : children;
        }

        const route = this.normalizeRoute(path, parent);

        if (children) {
            this.parseRoutes(children, { route, render });
        }

        this.routes[this.cleanPath(route)] = render;
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

    createRouter() {
        return function(location) {
            for (let route in this.routes) {
                const isMatch = this.match(route, location)
                const renderFunc = this.routes[route]

                if (isMatch) {
                    if (typeof renderFunc !== 'function') {
                        return renderFunc;
                    }
                    else {
                        return renderFunc() // return the render result
                    }
                }
            }

            return null
        }
    }

    match(path, pathname) {
        const regex = RegExp(path);
        const result = regex.exec(pathname);

        return !!result; // if there is a result, return true
    }

    render() {
        const { location } = this.props;

        return this.router(location);
    }
}