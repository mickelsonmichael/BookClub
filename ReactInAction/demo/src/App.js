import "./App.css";
import "../node_modules/bootstrap/dist/css/bootstrap.css";
import { BrowserRouter, Switch, Route, Link } from "react-router-dom";
import Chapter03 from "./chapter03";
import Chapter05 from "./chapter05";
import Chapter06 from "./chapter06";
import Chapter07 from "./chapter07";
import Chapter09 from "./chapter09";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <nav>
          <Link to="/">Home</Link>
          <Link to="/chapter-03">Chapter 03</Link>
          <Link to="/chapter-05">Chapter 05</Link>
          <Link to="/chapter-06">Chapter 06</Link>
          <Link to="/chapter-07">Chapter 07</Link>
          <Link to="/chapter-09">Chapter 09</Link>
        </nav>
        <main>
          <Switch>
            <Route exact path="/">
              <section>
                <p>Welcome to the Dev Book Club React App Demo! Or DBCRAP!</p>

                <p>Select a Chapter above to view some examples.</p>
              </section>
            </Route>

            <Route path="/chapter-03" component={Chapter03} />
            <Route path="/chapter-05" component={Chapter05} />
            <Route path="/chapter-06" component={Chapter06} />
            <Route path="/chapter-07" component={Chapter07} />
            <Route path="/chapter-09" component={Chapter09} />
          </Switch>
        </main>
      </BrowserRouter>
    </div>
  );
}

export default App;
