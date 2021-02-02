import "./App.css";
import "../node_modules/bootstrap/dist/css/bootstrap.css";
import Chapter05 from "./chapter05";
import { BrowserRouter, Switch, Route, Link } from "react-router-dom";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <nav>
          <Link to="/">Home</Link>
          <Link to="/chapter-05">Chapter 05</Link>
        </nav>
        <main>
          <Switch>
            <Route exact path="/">
              <p>Select a section above</p>
            </Route>
            <Route path="/chapter-05" component={Chapter05} />
          </Switch>
        </main>
      </BrowserRouter>
    </div>
  );
}

export default App;
